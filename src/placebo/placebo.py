from typing import Callable

import geopandas as gpd
import numpy as np
import pandas as pd
from src.constants import PROJECTED_CALIFORNIA, WGS84
from src.utils.logging_util import get_logger

logger = get_logger(__file__)

'''
Placebo test set is randomly generated sample of untreated (unburned) GEDI
shots.

Purpose:
    Placebo test set will be used to evaluate each counterfactual method. As
    such, for methods that require data callibration or training, placebo
    test set will be explicitly excluded from the training and callibration
    datasets. This allows us to prevent test overfitting and test for the
    generalizability of the method.

Sampling Requirements:
    We want to accomplish the following two goals with our sampling
    methodology:
    1.  Diminish the effects of spatial autocorrelation: GEDI shots will
        exhibit some level of spatial autocorrelation. As such, simple random
        sampling of untreated shots may be inadequate - as placebo test set and
        training test set may contain samples that are extremely similar to
        each other.
    2.  Placebo tests should mimic real-world scenario and our ultimate
        evaluation criteria. We want to understand which scenario works on
        wildfires. Wildfires always burn continous areas, and do not sample
        randomly. As such, we want our placebo tests to mimic the actual
        treatment patterns.

Implementation:
    Set out 20-25% of the total untreated units.
    Sample random locations in the ROI. For each random location, place all
    shots within radius X into the placebo test set. This circle represents a
    placebo fire. Radius X is a random sample from historic fire sizes.

'''


def create_placebo_test_set(
    untreated: gpd.GeoDataFrame,
    fires: gpd.GeoDataFrame,
    save_method: Callable = None
):
    logger.info("Converting all dataframes to projected CRS")
    untreated_proj = untreated.to_crs(PROJECTED_CALIFORNIA)
    historic_areas_proj = fires.to_crs(PROJECTED_CALIFORNIA).geometry.area

    logger.info("Creating placebo and callibration set.")
    placebo, callibration = _create_placebo_test_set(
        untreated_proj,
        historic_areas_proj
    )

    # Save.
    if save_method is not None:
        logger.info("Saving placebo and callibration sets.")
        save_method(placebo.to_crs(WGS84), callibration.to_crs(WGS84))

    return placebo, callibration


def _create_placebo_test_set(
    untreated: pd.DataFrame,
    historic_fire_sizes: pd.Series
):
    logger.info("Create placebo test - entry")
    placebo_set = pd.DataFrame()
    remaining = untreated.copy()

    # Step 1. Determine the size of the placebo test.
    min_bound = 0.2 * len(untreated)

    while (len(placebo_set) < min_bound):
        logger.info(f"Size of constructed placebo dataset: {len(placebo_set)}")
        # Step 2. Sample random location within ROI.
        center = remaining.sample().geometry.iloc[0]

        # Step 3. Sample random radius from historic fire size distribution.
        radius = np.sqrt(historic_fire_sizes.sample() / np.pi)
        logger.info(f"Placebo sample radius size: {radius}")

        # Step 4: Create a circle geometry representing the perimeter of the
        # placebo fire.
        circle = gpd.GeoDataFrame(
            geometry=center.buffer(radius),
            crs=PROJECTED_CALIFORNIA)

        # Step 5. Extract shots within the fake fire perimeter and place them
        # in the placebo test set.
        within = remaining.sjoin(
            circle,
            how="inner",
            predicate="within").drop(columns=["index_right"])

        placebo_set = pd.concat([placebo_set, within])
        remaining = remaining[~remaining.index.isin(within.index)]
        logger.info(f"Size of the updated placebo dataset: {len(placebo_set)}")
        logger.info(f"Size of the remaining dataset: {len(remaining)}")

    return placebo_set, remaining
