# Algo 4 - for finding control shots.
from src.data import fire_perimeters
from src.data import k_nn
import geopandas as gpd
import numpy as np
from src.data.clustering import cluster
import pandas as pd


def match_with_nearest_shot_from_terrain_cluster(
    fire: fire_perimeters.Fire,
    gedi: gpd.GeoDataFrame,
    buffer_size: int,
    num_samples: int,
    n_clusters: int,
    cluster_on_columns: list[str]
) -> gpd.GeoDataFrame:
    buffer = fire.get_buffer(buffer_size, 100)

    # Find unburned shots within the buffer.
    within_buffer = gedi.sjoin(
        buffer, how="inner", predicate="within")

    within_fire = gedi.sjoin(
        fire.fire, how="inner", predicate="within")

    # Cluster terrain on combined shots.
    combined_gdf = pd.concat([within_fire, within_buffer])

    clustered_df = cluster(combined_gdf, cluster_on_columns, n_clusters)

    # Separate them again into left and right groups.
    within_fire_clustered = pd.merge(clustered_df, within_fire, how='inner')
    within_buffer_clustered = pd.merge(
        clustered_df, within_buffer, how='inner')

    print('Enter clustering knn')
    processed = []
    # For each cluster, find the closest.
    for cluster_idx in range(n_clusters):
        fire_cluster = within_fire_clustered[within_fire_clustered.cluster == cluster_idx]
        buffer_cluster = within_buffer_clustered[within_buffer_clustered.cluster == cluster_idx]

        if fire_cluster.empty:
            continue

        match_indeces, _ = k_nn.nearest_neighbors(
            fire_cluster.to_crs(epsg=4326),
            buffer_cluster.to_crs(epsg=4326),
            min(num_samples, buffer_cluster.shape[0]))

        fire_cluster['agbd_control_mean'] = np.apply_along_axis(
            lambda x: buffer_cluster.iloc[x].agbd.mean(), 1, match_indeces)
        fire_cluster['agbd_control_median'] = np.apply_along_axis(
            lambda x: buffer_cluster.iloc[x].agbd.median(), 1, match_indeces)

        processed.append(fire_cluster)

    print(len(processed))
    return pd.concat(processed)
