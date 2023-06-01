import geopandas as gpd
import numpy as np
import pandas as pd
from src.data.k_nn import nn_control
from src.data.clustering import find_closest_in_cluster
from typing import Callable


def evaluate_control_on_untreated_pixels(
        untreated: gpd.GeoDataFrame,
        column: str,
        get_control_function: Callable):
    return evaluate_control(untreated, untreated, column, f'ref_{column}',
                            get_control_function)


def evaluate_control(
        left_gdf: gpd.GeoDataFrame,
        right_gdf: gpd.GeoDataFrame,
        in_column: str,
        out_column: str,
        get_control_function: Callable):
    match_output = get_control_function(left_gdf,
                                        right_gdf,
                                        in_column,
                                        out_column)

    in_column_vals = match_output.in_column.values
    out_column_vals = match_output.out_column.values

    rel_vals = np.divide(in_column_vals, out_column_vals)
    perfect_result = np.ones(rel_vals.shape)

    # Calculate RMSE.
    return r_mse(rel_vals, perfect_result)


'''
Below is a list of potential methods for finding a control. Each will be
evaluated separately.
'''


def closest_200_mean(left_gdf: gpd.GeoDataFrame,
                     right_gdf: gpd.GeoDataFrame,
                     in_column: str,
                     out_column: str):
    return nn_control(left_gdf, right_gdf, in_column, out_column,
                      lambda x: x.mean(), 200)


def closest_200_median(left_gdf: gpd.GeoDataFrame,
                       right_gdf: gpd.GeoDataFrame,
                       in_column: str,
                       out_column: str):
    return nn_control(left_gdf, right_gdf, in_column, out_column,
                      lambda x: x.median(), 200)


def closest_200_mean_on_clustered_terrain(left_gdf: gpd.GeoDataFrame,
                                          right_gdf: gpd.GeoDataFrame,
                                          in_column: str,
                                          out_column: str):
    return find_closest_in_cluster(left_gdf, right_gdf, in_column, out_column,
                                   ["elevation", "slope"], 5,
                                   lambda x: x.mean(), 200)


def r_mse(pred, y): return round(math.sqrt(((pred-y)**2).mean()), 6)
