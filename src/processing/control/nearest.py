# Algo 3 - for finding control shots.

from src.data import fire_perimeters
from src.data import k_nn
import geopandas as gpd
import numpy as np


def match_with_nearest_shot_per_pixel(
    fire: fire_perimeters.Fire,
    gedi: gpd.GeoDataFrame,
    buffer_size: int,
    num_samples: int
) -> gpd.GeoDataFrame:
    buffer = fire.get_buffer(buffer_size, 100)

    # Find unburned shots within the buffer.
    within_buffer = gedi.sjoin(
        buffer, how="inner", predicate="within")

    within_fire = gedi.sjoin(
        fire.fire, how="inner", predicate="within")

    match_indeces, _ = k_nn.nearest_neighbors(
        within_fire.to_crs(epsg=4326),
        within_buffer.to_crs(epsg=4326), num_samples)

    within_fire['agbd_control_mean'] = np.apply_along_axis(
        lambda x: within_buffer.iloc[x].agbd.mean(), 1, match_indeces)
    within_fire['agbd_control_median'] = np.apply_along_axis(
        lambda x: within_buffer.iloc[x].agbd.median(), 1, match_indeces)
    return within_fire
