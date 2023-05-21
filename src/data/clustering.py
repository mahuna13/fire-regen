from sklearn.cluster import KMeans, MiniBatchKMeans
from src.data import raster
import pandas as pd
import geopandas as gpd
import os
from threadpoolctl import threadpool_limits
from threadpoolctl import threadpool_info
from pprint import pprint

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUM_THREADS"] = "32"
os.environ["OMP_NUM_THREADS"] = "32"


def ecosystem_clustering(df, feature_cols):
    km = MiniBatchKMeans(n_clusters=5, verbose=1)
    km.fit(df[feature_cols])
    output = df.assign(cluster=km.predict(df[feature_cols]))

    return output


def filter_within_geometry(gdf1, query_gpd: gpd.GeoDataFrame):
    ''' Filter fire perimeters to be within geometries provided. '''
    return gdf1.sjoin(query_gpd, how="inner", predicate="within")


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

result = ecosystem_clustering(terrain_gedi_matched, ["elevation_mean"])

print('Save results')
result.to_csv(
    f'/maps/fire-regen/data/clustering/sierras_terrain_clustering_elevation_results.csv')
# print(result[['cluster', 'shot_number']])
