# Augments training or inference datasets with training features.
from pathlib import Path
from typing import Callable

import pandas as pd
from fastai.tabular.all import load_pickle, save_pickle
from src.constants import INTERMEDIATE_RESULTS, DATA_PATH
from src.data.processing import overlay
from src.utils.logging_util import get_logger
from src.data.adapters import mtbs
from src.placebo import placebo

logger = get_logger(__file__)

OUTPUT_PATH = f"{DATA_PATH}/analysis/recovery/rf/monthly"
INPUT_PATH = f"{INTERMEDIATE_RESULTS}/pipelines/recovery"


MONTHLY_LANDSAT_YEARS = [1985, 1988, 1993, 1998, 2003, 2008, 2013]
TTC_COLUMNS = ["tcc_2000", "tcc_2005", "tcc_2010", "tcc_2015"]


def add_landsat_monthly(
    df: pd.DataFrame,
    year: int
):
    monthly_overlays = list(
        Path(f"{overlay.MONTHLY_LANDSAT_FOLDER(year)}").iterdir())

    for filename in monthly_overlays:
        monthly_df = load_pickle(filename)
        cols = [col for col in monthly_df if (
            col.startswith("SR_") or col.startswith("NDVI"))]
        df_plus = df.join(monthly_df[cols], how="left")

    return df_plus


def add_tree_canopy_cover(
    df: pd.DataFrame
):
    tcc = load_pickle(overlay.TCC)
    df_plus = df.join(tcc[TTC_COLUMNS], how="left")
    return df_plus


def create_monthly_landsat_inference_data_sets(
    burned: pd.DataFrame,
    save_method: Callable = None
):
    logger.info("Creating inference data sets in 5 year buckets.")
    buckets = [5, 10, 15, 20, 25, 30, 35]
    all_sets = []
    for burn_bucket in buckets:
        inference_df = burned[burned.YSF_cat_5 == burn_bucket]
        landsat_year = year_for_monthly_landsat(burn_bucket)
        logger.info(
            f"Matching {burn_bucket} with landsat {landsat_year}.")

        # Add monthly landsat data to the dataset.
        inference_df = add_landsat_monthly(inference_df, landsat_year)

        all_sets.append(inference_df)

    result = pd.concat(all_sets)

    if save_method is not None:
        save_method(result)
    return result


def year_for_monthly_landsat(
    burn_bucket: int
):
    return max(1985, 2019 - burn_bucket - 1)


def save_placebo_set(path, placebo, calibration, set_name="set"):
    PLACEBO_FILE_NAME = f"placebo_{set_name}.pkl"
    CALIBRATION_FILE_NAME = f"calibration_{set_name}.pkl"

    save_pickle(f"{path}/{PLACEBO_FILE_NAME}", placebo)
    save_pickle(f"{path}/{CALIBRATION_FILE_NAME}", calibration)


if __name__ == '__main__':
    # Match burned.
    burned = load_pickle(f"{INPUT_PATH}/burned.pkl")
    burned_augmented = add_tree_canopy_cover(burned)
    burned_augmented = create_monthly_landsat_inference_data_sets(
        burned_augmented,
        lambda x: save_pickle(f"{OUTPUT_PATH}/burned.pkl", x)
    )

    # Match unburned.
    for year in MONTHLY_LANDSAT_YEARS:
        logger.info(f"Matching unburned for year {year}.")
        unburned = load_pickle(f"{INPUT_PATH}/unburned_lc_{year}.pkl")
        unburned = add_tree_canopy_cover(unburned)
        save_pickle(
            f"{OUTPUT_PATH}/unburned_{year}.pkl",
            add_landsat_monthly(unburned, year))

    # Create calibration and placebo datasets.
    sierra_fires = mtbs.get_mtbs_perimeters_for_sierras()
    for year in MONTHLY_LANDSAT_YEARS:
        logger.info(f"Creating calibration and placebo set for {year}.")
        unburned = load_pickle(f"{OUTPUT_PATH}/unburned_{year}.pkl")
        placebo.create_placebo_test_set(
            unburned,
            sierra_fires,
            lambda x, y: save_placebo_set(
                f"{OUTPUT_PATH}/{year}", x, y, "set_1")
        )
