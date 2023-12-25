# Module for fetching and processing calfire perimeter dataset.

# TODO: Add link for data location.

import geopandas as gpd
from fastai.tabular.all import save_pickle
from src.constants import DATA_PATH, USER_PATH
from src.data import fire_perimeters

# Fetch simplified regions of interest.
SEKI = gpd.read_file(f"{USER_PATH}/data/shapefiles/seki_convex_hull.shp")
SIERRAS = gpd.read_file(f"{USER_PATH}/data/shapefiles/sierras_convex_hull.shp")


def CALFIRE_BOUNDARY_BUFFER(distance):
    return f"{DATA_PATH}/calfire/perimeters_{distance}m_buffers.pkl"


def CALFIRE_PERIMETERS_TRIMMED(distance):
    return f"{DATA_PATH}/calfire/perimeters_{distance}m_trimmed.pkl"


CALFIRE_PERIMETERS_PATH = f"{USER_PATH}/data/fire_perimeters.gdb/"
CALFIRE_BOUNDARY_BUFFER_10m = CALFIRE_BOUNDARY_BUFFER(10)
CALFIRE_BURN_AREA_TRIMMED_10m = CALFIRE_PERIMETERS_TRIMMED(10)


def get_fire_perimeters_for_sierras():
    calfire_db = fire_perimeters.FirePerimetersDB(CALFIRE_PERIMETERS_PATH)
    return fire_perimeters.FirePerimeters(
        calfire_db).filter_for_region(SIERRAS)


def extract_buffers_around_perimeters(
    distance: int,
    save: bool = True
) -> gpd.GeoDataFrame:
    sierra_perimeters = get_fire_perimeters_for_sierras()
    original_crs = sierra_perimeters.perimeters.crs

    # Convert to a projected CRS.
    perimeters_projected = sierra_perimeters.perimeters.to_crs(epsg=3310)

    # Extract the buffers.
    boundary_buffers = perimeters_projected.boundary.buffer(distance)

    # Save.
    boundary_buffers_gdf = gpd.GeoDataFrame(
        perimeters_projected, geometry=boundary_buffers).to_crs(original_crs)

    if save:
        save_pickle(CALFIRE_BOUNDARY_BUFFER(distance), boundary_buffers_gdf)

    return boundary_buffers_gdf


# Trims the outside area around the fire, to exclude locations near the
# boundary to improve accuracy.
def trim_fire_area(
    distance: int,
    save: bool = True
) -> gpd.GeoDataFrame:
    sierra_perimeters = get_fire_perimeters_for_sierras()
    original_crs = sierra_perimeters.perimeters.crs

    # Convert to a projected CRS.
    perimeters_projected = sierra_perimeters.perimeters.to_crs(epsg=3310)

    # Extract the buffers around fire boundary.
    boundary_buffers = perimeters_projected.boundary.buffer(distance)

    # Trim area to exclude the area around fire perimeter.
    trimmed = perimeters_projected.difference(boundary_buffers)
    trimmed_gdf = gpd.GeoDataFrame(
        perimeters_projected, geometry=trimmed).to_crs(original_crs)

    if save:
        save_pickle(CALFIRE_PERIMETERS_TRIMMED(distance), trimmed_gdf)

    return trimmed_gdf
