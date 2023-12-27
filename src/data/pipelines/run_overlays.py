from src.constants import GEDI_INTERMEDIATE_PATH
from src.data.pipelines.extract_gedi_data import SIERRAS_GEDI_ID_COLUMNS
from src.data.processing import burn_areas_overlay as ba
from src.data.processing import burn_boundaries_overlay as bb
from src.data.processing import disturbance_overlays as da
from src.data.processing import raster_overlays
from src.data.processing import severity_overlay as se

OVERLAYS_PATH = f"{GEDI_INTERMEDIATE_PATH}/overlays"


def get_output_path(file_name: str):
    return f"{OVERLAYS_PATH}/{file_name}"

# TODO: Add Logging.
# TODO: Add checks not to run if files already exist.


if __name__ == '__main__':
    # Burn Datasets Overlays
    bb.overlay_with_boundary_buffers(
        SIERRAS_GEDI_ID_COLUMNS,
        get_output_path("burn_boundary_overlay.pkl"))

    ba.overlay_with_burn_area(SIERRAS_GEDI_ID_COLUMNS,
                              get_output_path("burn_area_overlay.pkl"))

    se.overlay_with_mtbs_fires(SIERRAS_GEDI_ID_COLUMNS,
                               get_output_path("mtbs_fires_overlay.pkl"))

    se.overlay_with_mtbs_dnbr(SIERRAS_GEDI_ID_COLUMNS,
                              get_output_path("mtbs_severity_overlay.pkl"))

    se.overlay_witg_mtbs_severity_categories(
        SIERRAS_GEDI_ID_COLUMNS,
        get_output_path("mtbs_severity_categories_overlay.pkl"))

    # Other Datasets Raster Overlays

    # Terrain Overlay
    raster_overlays.overlay_terrain(
        SIERRAS_GEDI_ID_COLUMNS, get_output_path("terrain_overlay.pkl"))

    # Landsat Overlay
    raster_overlays.overlay_landsat(
        SIERRAS_GEDI_ID_COLUMNS, get_output_path("landsat_overlay.pkl"))

    # Dynamic World Overlay
    raster_overlays.overlay_dynamic_world(
        SIERRAS_GEDI_ID_COLUMNS, get_output_path("dynamic_world_overlay.pkl"))

    # TODO: Add Land Cover

    # Disturbances Overlay
    da.overlay_with_disturbances(SIERRAS_GEDI_ID_COLUMNS,
                                 get_output_path("disturbances_overlay.pkl"))
