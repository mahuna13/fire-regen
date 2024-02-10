import os

import pandas as pd
from fastai.tabular.all import load_pickle, save_pickle
from src.data.pipelines.extract_gedi_data import (
    SEKI_GEDI_ID_COLUMNS, SIERRAS_GEDI_ID_COLUMNS)
from src.data.processing import all_fires_overlay as fa
from src.data.processing import burn_boundaries_overlay as bb
from src.data.processing import disturbance_overlays as da
from src.data.processing import advanced_landsat_overlay as alo
from src.data.processing import overlay
from src.data.processing import pre_fire_ndvi_overlay as pfno
from src.data.processing import raster_overlays
from src.data.processing import severity_overlay as se
from src.utils.logging_util import get_logger

logger = get_logger(__file__)


def LANDCOVER(year):
    return f"{overlay.OVERLAYS_PATH}/land_cover_overlay_{year}.pkl"


def get_output_path(file_name: str, seki: bool):
    if seki:
        return f"{overlay.OVERLAYS_PATH}/seki_{file_name}"
    return f"{overlay.OVERLAYS_PATH}/{file_name}"


def run_overlay(overlay_fn, file_name, override=False, seki=False):
    logger.info(f"Running overlay: {overlay_fn.__name__}")
    output_path = get_output_path(f"{file_name}", seki)
    if not override:
        if os.path.exists(output_path):
            logger.info(f"Overlay already exists: {output_path} \n")
            return  # file already exists, and we shouldn't override.

    gedi_shots = SEKI_GEDI_ID_COLUMNS if seki else SIERRAS_GEDI_ID_COLUMNS
    result = overlay_fn(load_pickle(gedi_shots))

    if result is None:
        logger.info("No overlay obtained.")
        return

    save_pickle(output_path, result)
    logger.info("Done! \n")
    return result


def run_overlay_for(df, overlay_fn, file_name, override=False, seki=False):
    logger.info(f"Running overlay: {overlay_fn.__name__}")
    output_path = get_output_path(f"{file_name}", seki)
    if not override:
        if os.path.exists(output_path):
            logger.info(f"Overlay already exists: {output_path} \n")
            return  # file already exists, and we shouldn't override.

    result = overlay_fn(df)
    save_pickle(output_path, result)
    logger.info("Done! \n")
    return result


def overlay_recent_land_cover():
    # Combine LC with dynamic world data, to provide two different sources for
    # land cover.

    # Load shots already combined with dynamic world.
    gedi_shots = load_pickle(overlay.DYNAMIC_WORLD)

    gedi_df_combined_years = []
    for year in range(2019, 2024):
        logger.info(f"Consolidate land cover for year {year}.")
        gedi_for_year = gedi_shots[gedi_shots.absolute_time.dt.year == year]

        lc_df = load_pickle(LANDCOVER(min(2021, year - 1)))

        joined_lc = gedi_for_year.join(
            lc_df[["land_cover_std", "land_cover_median"]], how="left")

        gedi_df_combined_years.append(joined_lc)

    result = pd.concat(gedi_df_combined_years)
    save_pickle(overlay.RECENT_LAND_COVER, result)


def overlay_land_cover(seki: bool):
    if seki:
        return

    START = 1985
    END = 2022

    for year in range(START, END):
        run_overlay(lambda x: raster_overlays.overlay_land_cover(x, year),
                    f"land_cover_overlay_{year}.pkl", seki=seki)


def overlay_all_landsat_years(seki: bool):
    for year in range(1984, 2023):
        run_overlay(
            lambda x: raster_overlays.overlay_landsat_for_year(x, year),
            f"landsat_overlay_{year}.pkl", seki=seki)


def run_all_overlays(seki=False):
    # Burn Datasets Overlays
    run_overlay(bb.overlay_with_boundary_buffers, "burn_boundary_overlay.pkl",
                seki=seki)

    run_overlay(fa.overlay_with_all_fires, "all_fires_overlay.pkl", seki=seki)

    run_overlay(lambda x:
                se.overlay_with_mtbs_fires(x,
                                           distance=100,
                                           post_fire_only=False),
                "mtbs_fires_overlay_all.pkl", seki=seki)

    run_overlay(lambda x: se.overlay_with_mtbs_fires(x, distance=100),
                "mtbs_fires_overlay.pkl", seki=seki)

    run_overlay(lambda x: se.overlay_with_mtbs_dnbr(
        x, distance=100, post_fire=False), "mtbs_severity_overlay_all.pkl",
        seki=seki)

    mtbs_overlay = run_overlay(lambda x: se.overlay_with_mtbs_dnbr(
        x, distance=100, post_fire=True), "mtbs_severity_overlay.pkl",
        seki=seki)

    run_overlay_for(mtbs_overlay, pfno.overlay_pre_fire_NDVI,
                    "mtbs_severity_overlay_with_ndvi.pkl",
                    seki=seki)

    run_overlay(se.overlay_with_mtbs_severity_categories,
                "mtbs_severity_categories_overlay.pkl", seki=seki)

    # Other Datasets Raster Overlays

    # Terrain Overlay
    run_overlay(raster_overlays.overlay_terrain,
                "terrain_overlay.pkl", seki=seki)

    # Landsat Overlay
    run_overlay(raster_overlays.overlay_landsat,
                "landsat_overlay.pkl", seki=seki)

    # NDVI Overlay
    run_overlay(raster_overlays.overlay_ndvi,
                "ndvi_timeseries_overlay.pkl", seki=seki)

    # Dynamic World Overlay
    run_overlay(raster_overlays.overlay_dynamic_world,
                "dynamic_world_overlay.pkl", seki=seki)

    # Land Cover
    overlay_land_cover(seki=seki)

    # Disturbances Overlay
    run_overlay(da.overlay_with_disturbances,
                "disturbances_overlay.pkl", seki=seki)

    # Adv Landsat
    for kind in ["mean", "min", "max", "stddev", "qt_25", "qt_50", "qt_75"]:
        run_overlay(lambda x: alo.overlay_advanced_landsat(x, 2019, kind),
                    f"advanced_landsat_overlay_{kind}.pkl", seki=seki)

    # Monthly Landsat
    year = 2003
    for month in range(1, 13):
        run_overlay(lambda x: alo.overlay_monthly_landsat(x, year, month),
                    f"monthly_landsat_overlay_{year}_{month}.pkl", seki=seki)

    # GFCC Overlay - tree canopy cover.
    run_overlay(raster_overlays.overlay_tree_cover,
                "tree_canopy_cover_overlay.pkl", seki=seki)

    overlay_all_landsat_years(seki)


if __name__ == '__main__':
    # TODO: add params to main
    seki = False
    run_all_overlays(seki)
