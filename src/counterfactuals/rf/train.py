from typing import Callable

import numpy as np
import pandas as pd
from fastai.tabular.all import (Categorify, FillMissing, TabularPandas,
                                cont_cat_split, IndexSplitter,
                                range_of)
from sklearn.ensemble import RandomForestRegressor
from src.utils.eval import r_squared, rmse, rma_regression
from src.utils.logging_util import get_logger

import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


logger = get_logger(__file__)


def train_rf(
    training_df: pd.DataFrame,
    dep_var: str,
    features: list[str],
    log: bool = False,
    save_func: Callable = None,
    test_df: pd.DataFrame = None
):
    training_df = training_df.copy()
    if test_df is not None:
        test_df = test_df.copy()

    if log:
        logger.debug("Optimizing for log")
        training_df = training_df[training_df[dep_var] != 0]
        training_df[dep_var] = np.log(training_df[dep_var])
        if test_df is not None:
            test_df = test_df[test_df[dep_var] != 0]
            test_df[dep_var] = np.log(test_df[dep_var])

    if test_df is not None:
        df = pd.concat([training_df, test_df]).reset_index()
        splits = IndexSplitter(
            df[df.shot_number.isin(test_df.index.values)].index
        )((range_of(df)))
    else:
        df = training_df.copy()
        splits = None

    to = prep_data_for_rf(df, features, dep_var, splits)

    xs, y = to.train.xs, to.train.y
    logger.debug("Start model training.")
    m = rf(xs, y)
    logger.debug("Training complete.\n")

    logger.info("Training Accuracy:")
    log_accuracy(y, m.predict(xs))

    logger.info(f"Validation error: {m.oob_score_}")

    if test_df is not None:
        logger.info("Test Accuracy:")
        log_accuracy(to.valid.y, m.predict(to.valid.xs))

    if save_func is not None:
        save_func(m, to)

    return m, to


def log_accuracy(y, y_pred):
    logger.info(f"RMSE: {rmse(y_pred, y)};")
    logger.info(f"R^2: {r_squared(y, y_pred)}")
    logger.info(f"RMA: {rma_regression(y, y_pred)}")


def prep_data_for_rf(
        df: pd.DataFrame,
        features: list[str],
        dep_var: str,
        splits=None):
    df_features = df[features + [dep_var]]
    logger.info("PREP PREP PREP--")
    procs = [Categorify, FillMissing]

    cont, cat = cont_cat_split(df_features, 1, dep_var=dep_var)
    return TabularPandas(df_features, procs, cat, cont, y_names=dep_var,
                         splits=splits)


def rf(
        xs,
        y,
        n_estimators=100,
        max_samples=0.85,
        max_features=0.5,
        min_samples_leaf=30,
        max_leaf_nodes=None,
        **kwargs):
    return RandomForestRegressor(
        n_jobs=-1,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        oob_score=True,
        max_leaf_nodes=max_leaf_nodes).fit(xs, y)
