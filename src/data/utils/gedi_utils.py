import geopandas as gpd
import numpy as np
import pandas as pd
from fastai.tabular.all import load_pickle
from src import constants

INDEX = 'shot_number'


def get_severity(df, severity):
    return df[df.severity == severity]


def get_gedi_shots(input_path: str) -> gpd.GeoDataFrame:
    gedi = load_pickle(input_path)

    gdf = gpd.GeoDataFrame(gedi,
                           geometry=gpd.points_from_xy(gedi.longitude,
                                                       gedi.latitude),
                           crs=constants.WGS84)
    gdf.set_index(INDEX, inplace=True)
    return gdf


def add_time_since_burn_categories(df: gpd.GeoDataFrame):
    c = pd.cut(
        df[['time_since_burn']].stack(),
        [-np.inf, 0, 10, 20, 30, np.inf],
        labels=['unburned', 'burn_10', 'burn_20', "burn_30", "burn_40"]
    )
    return df.join(c.unstack().add_suffix('_cat'))


def add_time_since_burn_categories_3(df: gpd.GeoDataFrame):
    c = pd.cut(
        df[['time_since_burn']].stack(),
        range(0, 37, 3),
        labels=['burn_3', 'burn_6', 'burn_9',
                "burn_12", "burn_15", "burn_18", "burn_21",
                "burn_24", "burn_27", "burn_30", "burn_33", "burn_36"]
    )
    return df.join(c.unstack().add_suffix('_cat_3'))


def add_time_since_burn_categories_5(df: gpd.GeoDataFrame):
    c = pd.cut(
        df[['time_since_burn']].stack(),
        range(0, 37, 5),
        labels=['burn_5', 'burn_10', 'burn_15',
                "burn_20", "burn_25", "burn_30", "burn_35"]
    )
    return df.join(c.unstack().add_suffix('_cat_5'))


def add_time_since_burn_categories_7(df: gpd.GeoDataFrame):
    c = pd.cut(
        df[['time_since_burn']].stack(),
        range(0, 37, 7),
        labels=['burn_7', 'burn_14', 'burn_21',
                "burn_28", "burn_35"]
    )
    return df.join(c.unstack().add_suffix('_cat_7'))


def add_time_since_burn_categories_10(df: gpd.GeoDataFrame):
    c = pd.cut(
        df[['time_since_burn']].stack(),
        range(0, 41, 10),
        labels=['burn_10', 'burn_20', "burn_30", "burn_40"]
    )
    return df.join(c.unstack().add_suffix('_cat_10'))


def print_burn_stats(df):
    # Gedi statistics
    unburned_ratio = (df[df.burn_counts_median == 0].shape[0])/df.shape[0]
    high_burn_ratio = (df[(df.burn_severity_median == 4)].shape[0])/df.shape[0]
    medium_burn_ratio = (
        df[(df.burn_severity_median == 3)].shape[0])/df.shape[0]
    low_burn_ratio = (df[(df.burn_severity_median == 2)].shape[0])/df.shape[0]

    print(f'Unburned ratio: {unburned_ratio*100}%')
    print(f'High-burned ratio: {high_burn_ratio*100}%')
    print(f'Medium-burned ratio: {medium_burn_ratio*100}%')
    print(f'Low-burned ratio: {low_burn_ratio*100}%')
