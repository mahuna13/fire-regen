# Augments training or inference datasets with training features.
from pathlib import Path
from typing import Callable

import pandas as pd
from fastai.tabular.all import load_pickle, save_pickle
from src.constants import DATA_PATH, INTERMEDIATE_RESULTS
from src.data.adapters import mtbs
from src.data.processing import overlay
from src.placebo import placebo
from src.utils.logging_util import get_logger

logger = get_logger(__file__)

OUTPUT_PATH = f"{DATA_PATH}/analysis/recovery/rf/monthly"
INPUT_PATH = f"{INTERMEDIATE_RESULTS}/pipelines/recovery"


MONTHLY_LANDSAT_YEARS = [1985, 1988, 1993, 1998, 2003, 2008, 2013]
TTC_COLUMNS = ["tcc_2000", "tcc_2005", "tcc_2010", "tcc_2015"]


def add_landsat_monthly(
    df: pd.DataFrame,
    year: int
):
    result = df.copy()
    monthly_overlays = list(
        Path(f"{overlay.MONTHLY_LANDSAT_FOLDER(year)}").iterdir())

    for filename in monthly_overlays:
        monthly_df = load_pickle(filename)
        cols = [col for col in monthly_df if (
            col.startswith("SR_") or col.startswith("NDVI"))]
        result = result.join(monthly_df[cols], how="left")

    return result


def add_tree_canopy_cover(
    df: pd.DataFrame
):
    tcc = load_pickle(overlay.TCC)
    return df.join(tcc[TTC_COLUMNS], how="left")


def add_recent_ndvi(
    df: pd.DataFrame
):
    ndvi = load_pickle(overlay.NDVI_RECENT)
    return df.join(ndvi, how="left")


def create_monthly_landsat_inference_data_sets(
    burned: pd.DataFrame,
    save_method: Callable
):
    logger.info("Creating inference data sets in 5 year buckets.")
    buckets = [5, 10, 15, 20, 25, 30, 35]
    for burn_bucket in buckets:
        inference_df = burned[burned.YSF_cat_5 == burn_bucket]
        landsat_year = year_for_monthly_landsat(burn_bucket)
        logger.info(
            f"Matching {burn_bucket} with landsat {landsat_year}.")

        # Add monthly landsat data to the dataset.
        inference_df = add_landsat_monthly(inference_df, landsat_year)

        save_method(inference_df, landsat_year)


def year_for_monthly_landsat(
    burn_bucket: int
):
    return max(1985, 2019 - burn_bucket - 1)


def save_placebo_set(path, placebo_ds, calibration_ds, k_fold="set"):
    PLACEBO_FILE_NAME = f"placebo_{k_fold}.pkl"
    CALIBRATION_FILE_NAME = f"calibration_{k_fold}.pkl"

    save_pickle(f"{path}/{PLACEBO_FILE_NAME}", placebo_ds)
    save_pickle(f"{path}/{CALIBRATION_FILE_NAME}", calibration_ds)


if __name__ == '__main__':
    # TODO: take argument through command line
    k_fold = "set_5"

    # Match burned.
    burned = load_pickle(f"{INPUT_PATH}/burned.pkl")
    burned = add_tree_canopy_cover(burned)
    burned = add_recent_ndvi(burned)
    create_monthly_landsat_inference_data_sets(
        burned,
        lambda x, year: save_pickle(f"{OUTPUT_PATH}/burned_{year}.pkl", x)
    )

    # Match unburned.
    for year in MONTHLY_LANDSAT_YEARS:
        logger.info(f"Matching unburned for year {year}.")
        unburned = load_pickle(f"{INPUT_PATH}/unburned_lc_{year}.pkl")
        unburned = add_tree_canopy_cover(unburned)
        unburned = add_recent_ndvi(unburned)
        save_pickle(
            f"{OUTPUT_PATH}/unburned_{year}.pkl",
            add_landsat_monthly(unburned, year))

    # Create calibration and placebo datasets.
    sierra_fires = mtbs.get_mtbs_perimeters_for_sierras()
    for year in MONTHLY_LANDSAT_YEARS:
        logger.info(f"Creating calibration and placebo set for {year}.")
        unburned = load_pickle(f"{OUTPUT_PATH}/unburned_{year}.pkl")
        small_df = unburned[["geometry"]]
        placebo_set, calibration_set = placebo.create_placebo_test_set(
            small_df,
            sierra_fires
        )
        placebo_full = unburned.loc[placebo_set.index]
        calibration_full = unburned.loc[calibration_set.index]
        save_placebo_set(
            f"{OUTPUT_PATH}/{year}",
            placebo_full,
            calibration_full,
            k_fold)
