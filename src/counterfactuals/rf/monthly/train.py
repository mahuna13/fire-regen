import pickle
import warnings
from pathlib import Path

from fastai.tabular.all import TabularPandas, load_pickle, patch, save_pickle
from src.constants import DATA_PATH
from src.counterfactuals.rf import rf
from src.utils.logging_util import get_logger

logger = get_logger(__file__)


def FILES_PATH(year):
    return f"{DATA_PATH}/analysis/recovery/rf/monthly/{year}"


def MODEL_PATH(year, k_fold, dep_var):
    return f"{FILES_PATH(year)}/model_{dep_var}_{k_fold}.pkl"


def TO_PATH(year, k_fold, dep_var):
    return f"{FILES_PATH(year)}/to_{dep_var}_{k_fold}.pkl"


MONTHLY_LANDSAT_YEARS = [1985, 1988, 1993, 1998, 2003, 2008, 2013]
TTC_COLUMNS = ["tcc_2000", "tcc_2005", "tcc_2010", "tcc_2015"]


def get_training_data(
        year: int,
        k_fold: str = "set"):
    PLACEBO_FILE_NAME = f"placebo_{k_fold}.pkl"
    CALIBRATION_FILE_NAME = f"calibration_{k_fold}.pkl"

    placebo = load_pickle(f"{FILES_PATH(year)}/{PLACEBO_FILE_NAME}")
    calibration = load_pickle(f"{FILES_PATH(year)}/{CALIBRATION_FILE_NAME}")
    return placebo, calibration


def get_tcc_features(
        year: int):
    if year == 2003:
        return ["tcc_2000"]
    elif year == 2008:
        return ["tcc_2000", "tcc_2005"]
    elif year == 2013:
        return ["tcc_2000", "tcc_2005", "tcc_2010"]
    else:
        return []


@patch
def export(self: TabularPandas, fname='export.pkl', pickle_protocol=2):
    "Export the contents of `self` without the items"
    old_to = self
    self = self.new_empty()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pickle.dump(self, open(Path(fname), 'wb'), protocol=pickle_protocol)
        self = old_to


def save_model(m, to, year, dep_var, k_fold):
    logger.info("Save model and data.")
    MODEL_PATH = f"{FILES_PATH(year)}/model_{dep_var}_{k_fold}.pkl"
    TO_PATH = f"{FILES_PATH(year)}/to_{dep_var}_{k_fold}.pkl"

    save_pickle(MODEL_PATH, m)
    to.export(TO_PATH)


if __name__ == '__main__':
    # TODO: pass these via command line arguments
    DEP_VAR = "agbd"
    K_FOLD = "set_5"

    for year in MONTHLY_LANDSAT_YEARS:
        logger.info(f"Training RF for year: {year}.")

        test_ds, calibration_ds = get_training_data(year, K_FOLD)
        trainer = rf.MonthlyLandsatRF(
            year,
            additional_features=get_tcc_features(year))

        m, to = trainer.train(DEP_VAR, calibration_ds, test_ds)

        save_model(m, to, year, DEP_VAR, K_FOLD)
