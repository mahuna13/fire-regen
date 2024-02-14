import pickle

import numpy as np
import pandas as pd
from fastai.tabular.all import load_pickle, save_pickle
from src.counterfactuals.rf.monthly import data_prep as dp
from src.counterfactuals.rf.monthly import train
from src.utils.logging_util import get_logger

logger = get_logger(__file__)


def run_ensemble_inference(
        dep_var: str,
        year: int,
        k_folds: list[str]):
    inference_df = load_pickle(f"{dp.OUTPUT_PATH}/burned_{year}.pkl")
    logger.info(f"Running inference for year {year} and {dep_var}.")

    num_k_folds = len(k_folds)
    y_predict = np.zeros((len(inference_df), num_k_folds))

    for k_fold_index in range(num_k_folds):
        k_fold = k_folds[k_fold_index]

        logger.info(f"Running inference for k-fold {k_fold}.")
        m = load_pickle(train.MODEL_PATH(year, k_fold, dep_var))
        to = pickle.load(open(train.TO_PATH(year, k_fold, dep_var), 'rb'))

        logger.info("Processing data.")
        to_new = to.train.new(inference_df)
        to_new.process()

        logger.info("Run model to predict counterfactual.")
        prediction = m.predict(to_new.train.xs)
        y_predict[:, k_fold_index] = prediction

    # Take the mean of all predictions.
    inference_df[f"{dep_var}_cf"] = y_predict.mean(axis=1)
    inference_df[f"{dep_var}_std"] = y_predict.std(axis=1)
    return inference_df


def run():
    # TODO: take in dep var through the command line.
    dep_var = "agbd"
    k_folds = ["set_1", "set_2", "set_3", "set_4", "set_5"]

    all_years = []
    for year in dp.MONTHLY_LANDSAT_YEARS:
        df = run_ensemble_inference(dep_var, year, k_folds)
        all_years.append(df)

    save_pickle(
        f"{dp.OUTPUT_PATH}/inference_{dep_var}.pkl",
        pd.concat(all_years))


if __name__ == '__main__':
    run()
