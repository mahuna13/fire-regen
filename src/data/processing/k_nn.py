import geopandas as gpd
import numpy as np
from sklearn.neighbors import BallTree
from typing import Callable


def nn_control(
        left_gdf: gpd.GeoDataFrame,
        right_gdf: gpd.GeoDataFrame,
        in_column: str,
        out_column: str,
        operator: Callable,
        k_n: int) -> gpd.GeoDataFrame:
    output_gdf = left_gdf.copy()

    match_indeces, _ = nearest_neighbors(left_gdf, right_gdf, k_n)

    output_gdf[out_column] = np.apply_along_axis(
        lambda x: operator(right_gdf.iloc[x][in_column]), 1, match_indeces)

    return output_gdf


def get_nearest(src_points, candidates, k_neighbors=1):
    """
    Find nearest neighbors for all source points from a set of candidate points
    """

    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric='haversine')

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Return indices and distances
    return (indices, distances)


def nearest_neighbors(left_gdf, right_gdf, k_neighbors=1):
    """
    For each point in left_gdf, find closest points in right GeoDataFrame and
    return their indeces and distances.

    NOTICE: Assumes that the input Points are in WGS84 projection (lat/lon).
    """

    if left_gdf.crs != "WGS84" or right_gdf.crs != "WGS84":
        raise Warning("Your dataframe is not in crs 'WGS84'")

    # Parse coordinates from points and insert them into a numpy array as
    # RADIANS.
    left_radians = lat_lon_to_radians(left_gdf)
    right_radians = lat_lon_to_radians(right_gdf)

    # Find the nearest points
    # -----------------------
    # closest ==> index in right_gdf that corresponds to the closest point
    # dist ==> distance between the nearest neighbors (in meters)

    closest, dist = get_nearest(
        src_points=left_radians, candidates=right_radians,
        k_neighbors=k_neighbors)

    # Convert to meters from radians
    earth_radius = 6371000  # meters
    dist = dist * earth_radius

    return closest, dist


def lat_lon_to_radians(gdf: gpd.GeoDataFrame):
    def geom_to_radians(geom):
        return (geom.x * np.pi / 180, geom.y * np.pi / 180)

    geom_col = gdf.geometry.name
    return np.array(gdf[geom_col].apply(geom_to_radians).to_list())
