from src.constants import GEDI_INTERMEDIATE_PATH
from src.utils.logging_util import get_logger
from fastai.tabular.all import load_pickle, save_pickle
from src.data.pipelines import run_overlays as overlays
import pandas as pd
import os
import sys

logger = get_logger(__file__)

OVERLAYS_PATH = f"{GEDI_INTERMEDIATE_PATH}/overlays"


def get_overlay(file_name: str):
    return load_pickle(f"{OVERLAYS_PATH}/{file_name}")


def get_output_path(file_name: str):
    return f"{OVERLAYS_PATH}/{file_name}"


def exclude_fire_boundaries(df: pd.DataFrame):
    # Get rid of all the shots that fell around the boundary of any fire.
    boundary_shots = get_overlay("burn_boundary_overlay.pkl")
    return df[~df.index.isin(boundary_shots.index)]


def filter_burn_areas(df: pd.DataFrame):
    # Get fires from call fires.
    all_fires = get_overlay("all_fires_overlay.pkl")

    # Eliminate all shots that fall within small and recent fires.
    small_recent = all_fires[(all_fires.Shape_Area_Acres < 1000)
                             & ((all_fires.YEAR_ >= 1984))]
    df = df[~df.index.isin(small_recent.index)]
    logger.info(
        f"Number of GEDI shots after filtering small fires: {len(df)}")

    # Eliminate all shots that burned 10 years before 1984.
    old_fires = all_fires[(all_fires.YEAR_ < 1984) & (all_fires.YEAR_ >= 1974)]
    df = df[~df.index.isin(old_fires.index)]
    logger.info(
        f"Number of GEDI shots after filtering old fires: {len(df)}")

    # burned, burned once, and unburned - report length.
    burned_once = df[df.fire_count == 1]
    burned_multiple = df[df.fire_count > 1]
    unburned = df[df.fire_count == 0]
    logger.info(
        f"Number of GEDI shots that burned once: {len(burned_once)}")
    logger.info(
        f"Number of GEDI shots that burned many times: {len(burned_multiple)}")
    logger.info(
        f"Number of GEDI shots that didn't burn: {len(unburned)}")

    return burned_once, burned_multiple, unburned


def filter_burned_based_on_land_cover(df: pd.DataFrame):
    start = 1985
    end = 2022

    all_years = []
    for year in range(start, end):
        lc_year = max(1985, year - 1)
        filtered = df[df.Ig_Year == year]
        lc_df = get_overlay(f"land_cover_overlay_{lc_year}.pkl")

        filtered = filtered.join(
            lc_df[["land_cover_std", "land_cover_median"]], how="left")
        filtered = filtered[(filtered.land_cover_std == 0)
                            & (filtered.land_cover_median == 1)]
        all_years.append(filtered.drop(
            columns=["land_cover_std", "land_cover_median"]))

    return pd.concat(all_years)


def filter_unburned_based_on_land_cover(df: pd.DataFrame):
    recent_lc = load_pickle(overlays.RECENT_LAND_COVER).drop(
        columns=["absolute_time"])

    joined_lc = df.join(recent_lc, how="left")
    filtered = joined_lc[(joined_lc.land_cover_std == 0)
                         & (joined_lc.land_cover_median == 1)]

    return recent_lc.loc[filtered.index]


if __name__ == '__main__':
    burned_path = get_output_path("burned_once.pkl")
    unburned_path = get_output_path("unburned.pkl")
    if os.path.exists(burned_path) and os.path.exists(unburned_path):
        sys.exit()

    # Load all GEDI shots overlayed with MTBS data.
    shots = get_overlay("mtbs_severity_overlay.pkl")
    logger.info(f"Total number of GEDI shots to process: {len(shots)}")

    # Get rid of all the shots that fell around the boundary of any fire.
    shots = exclude_fire_boundaries(shots)
    logger.info(f"Number of GEDI shots after filtering for areas around \
        fire boundaries: {len(shots)}")

    burned_once, burned_multiple, unburned = filter_burn_areas(shots)

    burned_once_lc = filter_burned_based_on_land_cover(burned_once)
    logger.info(
        f"Number of burned shots after filtering for land cover: \
            {len(burned_once)}")

    unburned_lc = filter_unburned_based_on_land_cover(unburned)
    logger.info(
        f"Number of unburned shots after filtering for land cover: \
            {len(unburned_lc)}")

    save_pickle(f"{OVERLAYS_PATH}/burned_once_lc.pkl", burned_once_lc)
    save_pickle(f"{OVERLAYS_PATH}/burned_once.pkl", burned_once)
    save_pickle(f"{OVERLAYS_PATH}/burned_multiple.pkl", burned_multiple)
    save_pickle(f"{OVERLAYS_PATH}/unburned.pkl", unburned)
    save_pickle(f"{OVERLAYS_PATH}/unburned_lc.pkl", unburned_lc)
