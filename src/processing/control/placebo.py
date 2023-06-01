import geopandas as gpd
import numpy as np
import pandas as pd
from src.data.k_nn import nn_control
from src.data.clustering import find_closest_in_cluster
from typing import Callable
import math


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
    return calculate_rmse(match_output[in_column].values,
                          match_output[out_column].values)


def calculate_rmse(array_1: np.array, array_2: np.array):
    rel_vals = np.divide(array_1, array_2)
    perfect_result = np.ones(rel_vals.shape)

    # Calculate RMSE.
    return r_mse(rel_vals, perfect_result)


def calc_rmse_from_matches(
        left_gdf: gpd.GeoDataFrame,
        right_gdf: gpd.GeoDataFrame,
        matched_indeces: np.array,
        column: str,
        operator: Callable) -> int:
    matched_vals = np.apply_along_axis(
        lambda x: operator(right_gdf.iloc[x][column]), 1, matched_indeces)

    return calculate_rmse(matched_vals, left_gdf[column].values)


def calc_rmse_from_matches_based_on_distance(
        left_gdf: gpd.GeoDataFrame,
        right_gdf: gpd.GeoDataFrame,
        matched_indeces: np.array,
        matched_distances: np.array,
        column: str,
        operator: Callable,
        max_distance: int) -> int:

    def calc_val_from_closest(x):
        indeces = x[0]
        distances = x[1]
        closest = indeces[np.where(distances < max_distance)]
        return operator(right_gdf.iloc[closest][column])

    stack = np.stack([matched_indeces, matched_distances], axis=1)
    matched_vals = np.apply_along_axis(calc_val_from_closest, 1, stack)
    return calculate_rmse(matched_vals, left_gdf[column].values)


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
