from sklearn.cluster import KMeans, MiniBatchKMeans
from src.data import raster
from src.data.k_nn import nn_control
import pandas as pd
import geopandas as gpd
import os
from threadpoolctl import threadpool_limits
from threadpoolctl import threadpool_info
from typing import Callable
from pprint import pprint


def find_closest_in_cluster(
        left_gdf: gpd.GeoDataFrame,
        right_gdf: gpd.GeoDataFrame,
        in_column: str,
        out_column: str,
        cluster_on_columns: list[str],
        n_clusters: int,
        operator: Callable,
        k_n: int) -> gpd.GeoDataFrame:
    combined_gdf = pd.concat([left_gdf, right_gdf])

    # Check if the DataFrame contains all clustering columns.
    for column in cluster_on_columns:
        if column not in combined_gdf.columns:
            raise Warning(
                "DataFrame does not contain all columns for clustering.")

    # Run K Means clustering on the combined GDF.
    clustered_df = cluster(combined_gdf, cluster_on_columns, n_clusters)

    # Separate them again into left and right groups.
    left_clustered = pd.merge(clustered_df, left_gdf, how='inner')
    right_clustered = pd.merge(clustered_df, left_gdf, how='inner')

    processed = []
    # For each cluster, find the closest.
    for cluster_idx in range(n_clusters):
        left_cluster = left_clustered[left_clustered.cluster == cluster_idx]
        right_cluster = right_clustered[right_clustered.cluster == cluster_idx]

        left_cluster_out = nn_control(left_cluster, right_cluster, in_column,
                                      out_column, operator, k_n)
        processed.append(left_cluster_out.copy())

    return pd.concat(processed)


def cluster(df, feature_cols, n_clusters=5, verbose=0):
    # Limit threads, without it BLAS causes segmentation error.
    # TODO: link to the documentation about this error.
    pprint(threadpool_info())
    threadpool_limits(limits=1)
    threadpool_limits(limits=1, user_api='blas')

    km = MiniBatchKMeans(n_clusters=n_clusters, verbose=verbose)
    km.fit(df[feature_cols])
    output = df.assign(cluster=km.predict(df[feature_cols]))

    return output


def filter_within_geometry(gdf1, query_gpd: gpd.GeoDataFrame):
    ''' Filter fire perimeters to be within geometries provided. '''
    return gdf1.sjoin(query_gpd, how="inner", predicate="within")


'''
TODO: Remove below code.
print('Create terrain raster')
terrain_raster = raster.RasterSampler(
    raster.TERRAIN_RASTER, raster.TERRAIN_BANDS)

print('Load GEDI trees')
gedi_trees = pd.read_csv(
    '/maps/fire-regen/data/gedi_sierras_burn_lc_matched_trees_trimmed.csv', index_col=0)

gedi_trees_gdf = gpd.GeoDataFrame(gedi_trees, geometry=gpd.points_from_xy(
    gedi_trees.lon_lowestmode, gedi_trees.lat_lowestmode), crs=4326)

seki = gpd.read_file("data/shapefiles/seki_convex_hull.shp")

seki_gedi = filter_within_geometry(gedi_trees_gdf, seki)

print('Match GEDI to raster')
terrain_gedi_matched = terrain_raster.sample_2x2(
    gedi_trees, 'lon_lowestmode', 'lat_lowestmode')

# result = ecosystem_clustering(gedi_trees.head(
#    10), ["slope_mean", "aspect_mean", "elevation_mean", "soil_median"])

print('Cluster')
pprint(threadpool_info())
threadpool_limits(limits=1)
threadpool_limits(limits=1, user_api='blas')
pprint(threadpool_info())
# result = ecosystem_clustering(terrain_gedi_matched, [
#    "slope_mean", "aspect_mean", "elevation_mean", "soil_median"])

result = cluster(terrain_gedi_matched, ["elevation_mean"])

print('Save results')
result.to_csv(
    f'/maps/fire-regen/data/clustering/sierras_terrain_clustering_elevation_results.csv')
# print(result[['cluster', 'shot_number']])
'''
