import geopandas as gpd
from fastai.tabular.all import load_pickle
from src import constants

INDEX = 'shot_number'


def get_gedi_shots(input_path: str) -> gpd.GeoDataFrame:
    gedi = load_pickle(input_path)

    gdf = gpd.GeoDataFrame(gedi,
                           geometry=gpd.points_from_xy(gedi.longitude,
                                                       gedi.latitude),
                           crs=constants.WGS84)
    gdf.set_index(INDEX, inplace=True)
    return gdf
