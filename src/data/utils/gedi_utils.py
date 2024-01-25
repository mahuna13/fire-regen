import geopandas as gpd
import numpy as np
import pandas as pd
from fastai.tabular.all import load_pickle
from src import constants


def get_gedi_shots(input_path: str, index: str) -> gpd.GeoDataFrame:
    gedi = load_pickle(input_path)
    gdf = convert_to_geo_df(gedi)
    gdf.set_index(index, inplace=True)
    return gdf


def convert_to_geo_df(df: pd.DataFrame):
    return gpd.GeoDataFrame(df,
                            geometry=gpd.points_from_xy(df.longitude,
                                                        df.latitude),
                            crs=constants.WGS84)


def add_YSF_categories(df: pd.DataFrame, period: int):
    max_range = 40
    bins = [-np.inf, -1] + list(range(period, max_range, period)) + [np.inf]
    labels = [-1] + list(range(period, max_range, period)) + [max_range]

    c = pd.cut(df[['YSF']].stack(), bins, labels=labels)
    return df.join(c.unstack().add_suffix(f'_cat_{period}'))
