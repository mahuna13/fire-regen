from typing import Callable

import geopandas as gpd
from src.utils.logging_util import get_logger
from src.counterfactuals.rf import augment_data as augment

logger = get_logger(__file__)


def create_monthly_landsat_inference_data_sets(
    burned: gpd.GeoDataFrame,
    save_method: Callable = None
):
    logger.info("Creating inference data sets in 5 year buckets.")
    buckets = [5, 10, 15, 20, 25, 30, 35]
    all_sets = []
    for burn_bucket in buckets:
        inference_df = burned[burned.YSF_cat_5 == burn_bucket]

        # Add monthly landsat data to the dataset.
        inference_df = augment.add_landsat_monthly(
            inference_df,
            year_for_monthly_landsat(burn_bucket))

        if save_method is not None:
            save_method(inference_df, burn_bucket)

        all_sets.append(inference_df)
    return all_sets


def year_for_monthly_landsat(
    burn_bucket: int
):
    return max(1985, 2019 - burn_bucket - 1)
