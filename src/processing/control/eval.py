
from src.constants import DATA_PATH, USER_PATH
from src.data import fire_perimeters
import geopandas as gpd
import pandas as pd
import random
import math
from sklearn.metrics import *
import numpy as np
from typing import Callable


def evaluate_control(
    num_fires: int,
    gedi: gpd.GeoDataFrame,
    buffer_size: int,
    num_samples: int,
    func_to_eval: Callable,
    debug: bool = False
):
    # Assumes everything is converted to 3310 projection.
    fire_sizes = pd.read_csv(
        f"{DATA_PATH}/controls/fire_areas_1000_to_50000.csv", index_col=0).Shape_Area.values

    vals = []
    vals_controls_mean = []
    vals_controls_median = []
    for i in range(num_fires):
        fake_fire_size = random.choice(fire_sizes)
        if debug:
            return evaluate_control_on_single_fake_fire(
                fake_fire_size, gedi, buffer_size, num_samples, func_to_eval,
                debug)

        x, y, z = evaluate_control_on_single_fake_fire(
            fake_fire_size, gedi, buffer_size, num_samples, func_to_eval)

        vals.append(x)
        vals_controls_mean.append(y)
        vals_controls_median.append(z)

    vals = np.concatenate(vals)
    vals_controls_mean = np.concatenate(vals_controls_mean)
    vals_controls_median = np.concatenate(vals_controls_median)

    return np.array([rmse(vals, vals_controls_mean),
                     r_squared(vals, vals_controls_mean),
                     rmse(vals, vals_controls_median),
                     r_squared(vals, vals_controls_median)])


def evaluate_control_on_single_fake_fire(
    fake_fire_size: int,
    gedi: gpd.GeoDataFrame,
    buffer_size: int,
    num_samples: int,
    func_to_eval: Callable,
    debug: bool = False
):
    # Assumes everything is converted to 3310 projection.

    fake_fire_center = gedi.sample().geometry.iloc[0]
    fake_fire_radius = math.sqrt(fake_fire_size / math.pi)
    fake_fire_circle = fake_fire_center.buffer(fake_fire_radius)

    fake_fire = fire_perimeters.Fire(gpd.GeoDataFrame(geometry=gpd.GeoSeries(
        fake_fire_circle), columns=['ALARM_DATE', 'CONT_DATE'], crs=3310))

    vals_with_controls = func_to_eval(
        fake_fire, gedi, buffer_size, num_samples)
    if debug:
        return vals_with_controls

    vals = vals_with_controls.agbd.values
    vals_controls_mean = vals_with_controls.agbd_control_mean.values
    vals_controls_median = vals_with_controls.agbd_control_median.values

    return vals, vals_controls_mean, vals_controls_median


def rmse(samples, controls):
    if samples is None or controls is None:
        return None
    return math.sqrt(mean_squared_error(samples, controls))


def r_squared(samples, controls):
    if samples is None or controls is None:
        return None
    return r2_score(samples, controls)
