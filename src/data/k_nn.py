import geopandas as gpd
from sklearn.neighbors import BallTree
import numpy as np


def get_nearest(src_points, candidates, k_neighbors=1):
    """
    Find nearest neighbors for all source points from a set of candidate points
    """

    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric='haversine')

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    # Get closest indices and distances
    # Array at index 0 is the list of closest points to each src_point
    # Array at index 1 is the list of second closest points, etc.
    closest = indices
    closest_dist = distances

    # Return indices and distances
    return (closest, closest_dist)


def nearest_neighbors(left_gdf, right_gdf, column_to_avg, k_neighbors=1):
    """
    For each point in left_gdf, find closest points in right GeoDataFrame and 
    return their indeces and distances.

    NOTICE: Assumes that the input Points are in WGS84 projection (lat/lon).
    """

    # Ensure that index in right gdf is formed of sequential numbers
    right = right_gdf.copy().reset_index(drop=True)

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

    avg_column_from_nn = np.apply_along_axis(
        lambda x: right.loc[x][column_to_avg].mean(), 1, closest.T)

    # Convert to meters from radians
    earth_radius = 6371000  # meters
    dist = dist * earth_radius

    return closest.T, dist.T, avg_column_from_nn


def lat_lon_to_radians(gdf: gpd.GeoDataFrame):
    def geom_to_radians(geom):
        return (geom.x * np.pi / 180, geom.y * np.pi / 180)

    geom_col = gdf.geometry.name
    return np.array(gdf[geom_col].apply(geom_to_radians).to_list())