import os

import pandas as pd
from fastai.tabular.all import load_pickle, save_pickle
from src.constants import GEDI_INTERMEDIATE_PATH
from src.data.pipelines.extract_gedi_data import SIERRAS_GEDI_ID_COLUMNS
from src.data.processing import all_fires_overlay as fa
from src.data.processing import burn_boundaries_overlay as bb
from src.data.processing import disturbance_overlays as da
from src.data.processing import raster_overlays
from src.data.processing import severity_overlay as se
from src.utils.logging_util import get_logger

logger = get_logger(__file__)

OVERLAYS_PATH = f"{GEDI_INTERMEDIATE_PATH}/overlays"

# TODO: put file names for each overlay into constants accessible elsewhere.
DYNAMIC_WORLD = f"{OVERLAYS_PATH}/dynamic_world_overlay.pkl"
RECENT_LAND_COVER = f"{OVERLAYS_PATH}/recent_land_cover.pkl"


def LANDCOVER(year):
    return f"{OVERLAYS_PATH}/land_cover_overlay_{year}.pkl"


def get_output_path(file_name: str):
    return f"{OVERLAYS_PATH}/{file_name}"


def run_overlay(overlay_fn, file_name, override=False):
    logger.info(f"Running overlay: {overlay_fn.__name__}")
    output_path = get_output_path(file_name)
    if not override:
        if os.path.exists(output_path):
            logger.info(f"Overlay already exists: {output_path} \n")
            return  # file already exists, and we shouldn't override.

    overlay_fn(SIERRAS_GEDI_ID_COLUMNS, output_path)
    logger.info("Done! \n")


def overlay_recent_land_cover():
    # Combine LC with dynamic world data, to provide two different sources for
    # land cover.

    # Load shots already combined with dynamic world.
    gedi_shots = load_pickle(DYNAMIC_WORLD)

    gedi_df_combined_years = []
    for year in range(2019, 2024):
        logger.info(f"Consolidate land cover for year {year}.")
        gedi_for_year = gedi_shots[gedi_shots.absolute_time.dt.year == year]

        lc_df = load_pickle(LANDCOVER(min(2021, year - 1)))

        joined_lc = gedi_for_year.join(
            lc_df[["land_cover_std", "land_cover_median"]], how="left")

        gedi_df_combined_years.append(joined_lc)

    overlay = pd.concat(gedi_df_combined_years)
    save_pickle(RECENT_LAND_COVER, overlay)


if __name__ == '__main__':
    # Burn Datasets Overlays
    run_overlay(bb.overlay_with_boundary_buffers, "burn_boundary_overlay.pkl")

    run_overlay(fa.overlay_with_all_fires, "all_fires_overlay.pkl")

    run_overlay(lambda x, y:
                se.overlay_with_mtbs_fire_and_save(x,
                                                   y,
                                                   distance=30,
                                                   post_fire_only=False),
                "mtbs_fires_overlay_all.pkl")

    run_overlay(lambda x, y:
                se.overlay_with_mtbs_fire_and_save(x,
                                                   y,
                                                   distance=30),
                "mtbs_fires_overlay.pkl")

    run_overlay(lambda x, y: se.overlay_with_mtbs_dnbr(
        x, y, distance=30, post_fire=False), "mtbs_severity_overlay_all.pkl")

    run_overlay(lambda x, y: se.overlay_with_mtbs_dnbr(
        x, y, distance=30, post_fire=True), "mtbs_severity_overlay.pkl")

    run_overlay(se.overlay_with_mtbs_severity_categories,
                "mtbs_severity_categories_overlay.pkl")

    # Other Datasets Raster Overlays

    # Terrain Overlay
    run_overlay(raster_overlays.overlay_terrain, "terrain_overlay.pkl")

    # Landsat Overlay
    run_overlay(raster_overlays.overlay_landsat, "landsat_overlay.pkl")

    # Dynamic World Overlay
    run_overlay(raster_overlays.overlay_dynamic_world,
                "dynamic_world_overlay.pkl")

    # TODO: Add Land Cover

    # Disturbances Overlay
    run_overlay(da.overlay_with_disturbances, "disturbances_overlay.pkl")
