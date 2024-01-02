import os

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
