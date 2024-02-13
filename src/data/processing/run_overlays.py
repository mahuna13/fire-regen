import os

import pandas as pd
from fastai.tabular.all import load_pickle, save_pickle
from src.data.pipelines.extract_gedi_data import SIERRAS_GEDI_ID_COLUMNS
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


def run_overlay(
        overlay_fn,
        output_path,
        df=None,
        override=False):
    logger.info(f"Running overlay: {overlay_fn.__name__}")
    if not override:
        if os.path.exists(output_path):
            logger.info(f"Overlay already exists: {output_path} \n")
            return  # file already exists, and we shouldn't override.

    if df is None:
        df = load_pickle(SIERRAS_GEDI_ID_COLUMNS)

    result = overlay_fn(df)
    if result is None:
        logger.info("No overlay obtained.")
        return

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

        lc_df = load_pickle(overlay.LANDCOVER(min(2021, year - 1)))

        joined_lc = gedi_for_year.join(
            lc_df[["land_cover_std", "land_cover_median"]], how="left")

        gedi_df_combined_years.append(joined_lc)

    result = pd.concat(gedi_df_combined_years)
    save_pickle(overlay.RECENT_LAND_COVER, result)


def overlay_land_cover():

    START = 1985
    END = 2022

    for year in range(START, END):
        run_overlay(
            lambda x: raster_overlays.overlay_land_cover(x, year),
            overlay.LANDCOVER(year))


def overlay_all_landsat_years():
    for year in range(1984, 2023):
        run_overlay(
            lambda x: raster_overlays.overlay_landsat_for_year(x, year),
            overlay.LANDSAT(year))


def run_all_overlays():
    # Burn Datasets Overlays
    run_overlay(bb.overlay_with_boundary_buffers, overlay.MTBS_BURN_BOUNDARIES)
    run_overlay(fa.overlay_with_all_fires, overlay.ALL_CALFIRE_FIRES)

    run_overlay(lambda x:
                se.overlay_with_mtbs_fires(x,
                                           distance=100,
                                           post_fire_only=False),
                overlay.ALL_MTBS_FIRES)

    run_overlay(lambda x: se.overlay_with_mtbs_fires(x, distance=100),
                overlay.MTBS_FIRES)

    run_overlay(lambda x: se.overlay_with_mtbs_dnbr(
        x, distance=100, post_fire=False), overlay.ALL_MTBS_FIRES_SEVERITY)

    mtbs_overlay = run_overlay(lambda x: se.overlay_with_mtbs_dnbr(
        x, distance=100, post_fire=True), overlay.MTBS_FIRES_SEVERITY)

    run_overlay(pfno.overlay_pre_fire_NDVI,
                overlay.MTBS_SEVERITY_WITH_PREFIRE_NDVI, df=mtbs_overlay)

    run_overlay(se.overlay_with_mtbs_severity_categories,
                overlay.MTBS_SEVERITY_CATEGORIES)

    # Other Datasets Raster Overlays

    # Terrain Overlay
    run_overlay(raster_overlays.overlay_terrain, overlay.TERRAIN)

    # Landsat Overlay
    run_overlay(raster_overlays.overlay_landsat, overlay.RECENT_LANDSAT)

    # NDVI Overlay
    run_overlay(raster_overlays.overlay_ndvi, overlay.NDVI_TIMESERIES)

    # Dynamic World Overlay
    run_overlay(raster_overlays.overlay_dynamic_world, overlay.DYNAMIC_WORLD)

    # Land Cover
    overlay_land_cover()

    # Disturbances Overlay
    run_overlay(da.overlay_with_disturbances, overlay.DISTURBANCE_AGENTS)

    # Adv Landsat
    for kind in ["mean", "min", "max", "stddev", "qt_25", "qt_50", "qt_75"]:
        run_overlay(lambda x: alo.overlay_advanced_landsat(x, 2019, kind),
                    overlay.ADVANCED_LANDSAT(kind))

    # Monthly Landsat
    year = 1985
    for month in range(1, 13):
        run_overlay(lambda x: alo.overlay_monthly_landsat(x, year, month),
                    overlay.MONTHLY_LANDSAT(year, month))

    # GFCC Overlay - tree canopy cover.
    run_overlay(raster_overlays.overlay_tree_cover, overlay.TCC)

    overlay_all_landsat_years()


if __name__ == '__main__':
    # TODO: add params to main
    run_all_overlays()
