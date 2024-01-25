from src.constants import INTERMEDIATE_RESULTS
from src.utils.logging_util import get_logger
from fastai.tabular.all import load_pickle, save_pickle
from src.data.processing import overlay
import pandas as pd

pd.options.mode.chained_assignment = None
logger = get_logger(__file__)

PIPELINES_PATH = f"{INTERMEDIATE_RESULTS}/pipelines"


def get_overlay(file_name: str):
    return load_pickle(f"{overlay.OVERLAYS_PATH}/{file_name}")


def get_output_path(file_name: str):
    return f"{PIPELINES_PATH}/{file_name}"


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
        filtered = df[df.fire_ig_date.dt.year == year]
        lc_df = get_overlay(f"land_cover_overlay_{lc_year}.pkl")

        filtered = filtered.join(
            lc_df[["land_cover_std", "land_cover_median"]], how="left")
        filtered = filtered[(filtered.land_cover_std == 0)
                            & (filtered.land_cover_median == 1)]
        all_years.append(filtered.drop(
            columns=["land_cover_std", "land_cover_median"]))

    return pd.concat(all_years)


def filter_unburned_based_on_land_cover(df: pd.DataFrame):
    recent_lc = load_pickle(overlay.RECENT_LAND_COVER).drop(
        columns=["absolute_time"])

    joined_lc = df.join(recent_lc, how="left")
    filtered = joined_lc[(joined_lc.land_cover_std == 0)
                         & (joined_lc.land_cover_median == 1)]

    return df.loc[filtered.index]


def filter_burn_severity(df: pd.DataFrame):
    INVALID = -5000

    # Filter invalid dnbr values.
    df = df[df.dnbr_min > INVALID]
    return df


def save_burned_output(df: pd.DataFrame, file_name: str, prefix: str = ""):
    # Save output in two formats:
    # 1. Keeping only the fire info.
    # 2. Keeping all the fire columns, as that's relevant for severity analysis

    # Drop all severity columns and save only fire info.
    df_no_severity = df.drop(columns=['dNBR_offst', 'dNBR_stdDv', 'Low_T',
                                      'Mod_T', 'High_T', 'dnbr_mean',
                                      'dnbr_std', 'dnbr_median', 'dnbr_min',
                                      'dnbr_max', 'Low_T_adj', 'Mod_T_adj',
                                      'High_T_adj'])

    save_pickle(f"{PIPELINES_PATH}/{prefix}{file_name}", df_no_severity)

    logger.info(f"Number of GEDI shots before severity filtering: {len(df)}")
    df_severity = filter_burn_severity(df)
    logger.info(f"Number of GEDI shots afer severity filtering: \
                {len(df_severity)}")
    save_pickle(f"{PIPELINES_PATH}/{prefix}severity_{file_name}", df_severity)


def save_unburned_output(df: pd.DataFrame, file_name: str, prefix: str = ""):
    # Save output in two formats:
    # 1. Keeping only the fire info.
    # 2. Keeping all the fire columns, as that's relevant for severity analysis

    # Drop all severity columns and fire columns.
    df = df.drop(columns=['dNBR_offst', 'dNBR_stdDv', 'Low_T', 'Mod_T',
                          'High_T', 'dnbr_mean', 'dnbr_std', 'dnbr_median',
                          'Low_T_adj', 'Mod_T_adj', 'High_T_adj', "fire_id",
                          "fire_size_acres", "fire_name", "fire_ig_date",
                          "days_since_fire", "pre_fire_ndvi", 'dnbr_max',
                          'dnbr_min'])

    save_pickle(f"{PIPELINES_PATH}/{prefix}{file_name}", df)


def run(gedi_path: str, prefix: str):
    # Load all GEDI shots overlayed with MTBS data.
    shots = get_overlay(gedi_path)
    logger.info(f"Total number of GEDI shots to process: {len(shots)}")

    # Get rid of all the shots that fell around the boundary of any fire.
    shots = exclude_fire_boundaries(shots)
    logger.info(f"Number of GEDI shots after filtering for areas around \
        fire boundaries: {len(shots)}")

    burned_once, burned_multiple, unburned = filter_burn_areas(shots)
    save_burned_output(burned_once, "burned_once.pkl", prefix)
    save_burned_output(burned_multiple, "burned_multiple.pkl", prefix)
    save_unburned_output(unburned, "unburned.pkl", prefix)

    burned_once_lc = filter_burned_based_on_land_cover(burned_once)
    logger.info(
        f"Number of burned shots after filtering for land cover: \
            {len(burned_once_lc)}")
    save_burned_output(burned_once_lc, "burned_once_lc.pkl", prefix)

    unburned_lc = filter_unburned_based_on_land_cover(unburned)
    logger.info(
        f"Number of unburned shots after filtering for land cover: \
            {len(unburned_lc)}")
    save_unburned_output(unburned_lc, "unburned_lc.pkl", prefix)


if __name__ == '__main__':
    # gedi_path = "seki_mtbs_severity_overlay.pkl"
    gedi_path = "mtbs_severity_overlay_with_ndvi.pkl"
    # prefix = "seki_"
    # run(gedi_path, prefix="seki_")
    run(gedi_path, prefix="")
