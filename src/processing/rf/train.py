from fastai.tabular.all import load_pickle, save_pickle, patch
from src.constants import DATA_PATH
from src.data import gedi_raster_matching
from src.processing.rf import split_data
from fastai.tabular.all import *
from sklearn.metrics import *
from src.utils.logging_util import get_logger
from sklearn.ensemble import RandomForestRegressor
import numpy as np


logger = get_logger(__file__)


def r_mse(pred, y): return round(math.sqrt(mean_squared_error(y, pred)), 6)


def m_rmse(m, xs, y): return r_mse(m.predict(xs), y)


def r_squared(pred, y): return r2_score(y, pred)


def m_r2(m, xs, y): return r_squared(m.predict(xs), y)


def train_rf(year, geometry, save: bool = True, log: bool = False):
    # Get data for this year.
    logger.debug("Load training data from a pickle file.")
    gedi = load_pickle(f"{DATA_PATH}/rf/unburned/gedi_match_{year}.pkl")

    landsat_timeseries_legth = min(5, year - 1984)

    landsat_columns = [f"{x}_{y}"
                       for y in range(year - landsat_timeseries_legth, year)
                       for x in gedi_raster_matching.get_landsat_bands(y)]
    columns_to_use = ['agbd', 'pft_class', 'elevation',
                      'slope', 'aspect', 'soil', 'gedi_year'] + landsat_columns

    # Split data into training and validation.
    logger.debug("Split data into training and testing.")
    gedi_prepared = split_data.spatial_split_train_and_test_data(
        gedi, geometry, 5000)

    dep_var = 'agbd'
    if log:
        gedi_prepared = gedi_prepared[gedi_prepared.agbd != 0]
        gedi_prepared[dep_var] = np.log(gedi_prepared[dep_var])

    logger.debug("Prepare data for training.")
    df = gedi_prepared[columns_to_use]
    procs = [Categorify, FillMissing, Normalize]

    cont, cat = cont_cat_split(df, 1, dep_var=dep_var)
    splits = IndexSplitter(
        gedi_prepared[gedi_prepared.dataset == "test"].index)((range_of(df)))
    to = TabularPandas(df, procs, cat, cont, y_names=dep_var, splits=splits)

    xs, y = to.train.xs, to.train.y
    valid_xs, valid_y = to.valid.xs, to.valid.y

    logger.debug("Start model training.")
    m = rf(xs, y)
    logger.debug("Training complete.")

    rmse_train = m_rmse(m, xs, y)
    rmse_test = m_rmse(m, valid_xs, valid_y)
    r2_train = m_r2(m, xs, y)
    r2_test = m_r2(m, valid_xs, valid_y)

    logger.info(f"Year {year} - Training rmse: {rmse_train}; R^2: {r2_train}")
    logger.info(f"Year {year} - Validation error: {m.oob_score_}")
    logger.info(f"Year {year} - Test rmse: {rmse_test}; R^2: {r2_test}")

    if save:
        logger.debug(f"Save RF model for inference.")
        if log:
            save_pickle(f"{DATA_PATH}/rf/models_log/model_{year}.pkl", m)
            logger.debug(f"Save TabularPandas object for inference.")
            to.export(f"{DATA_PATH}/rf/models_log/to_{year}.pkl")
        else:
            save_pickle(f"{DATA_PATH}/rf/models/model_{year}.pkl", m)
            logger.debug(f"Save TabularPandas object for inference.")
            to.export(f"{DATA_PATH}/rf/models/to_{year}.pkl")

    return r2_train, m.oob_score_, r2_test, rmse_train, rmse_test


def rf(xs, y, n_estimators=100, max_samples=0.85,
       max_features=0.5, min_samples_leaf=30, max_leaf_nodes=None, **kwargs):
    return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators,
                                 max_samples=max_samples, max_features=max_features,
                                 min_samples_leaf=min_samples_leaf, oob_score=True, max_leaf_nodes=max_leaf_nodes).fit(xs, y)


@patch
def export(self: TabularPandas, fname='export.pkl', pickle_protocol=2):
    "Export the contents of `self` without the items"
    old_to = self
    self = self.new_empty()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pickle.dump(self, open(Path(fname), 'wb'), protocol=pickle_protocol)
        self = old_to
