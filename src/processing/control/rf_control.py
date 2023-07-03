import geopandas as gpd
from src.data import fire_perimeters
from fastai.tabular.all import load_pickle
from src.constants import DATA_PATH
import pickle


def match_with_rf_synthetic_estimate(
    fire: fire_perimeters.Fire,
    gedi: gpd.GeoDataFrame,
    buffer_size: int,
    num_samples: int,
    year: int
) -> gpd.GeoDataFrame:
    # In this algorithm, we will not be using shots from the fire perimeter as
    # control. Instead, carbon value comes from the pre-trained RF. For the
    # purposes of testing control, we will use the RF trained for 2019 fires.

    # Load RF model from 2019.
    m = load_pickle(f"{DATA_PATH}/rf/models/model_{year}.pkl")
    to = pickle.load(open(f"{DATA_PATH}/rf/models/to_{year}.pkl", 'rb'))

    print("matching shots within fire")
    # within_fire = gedi.sjoin(
    #    fire.fire, how="inner", predicate="within")
    print("Processing data")
    to_new = to.train.new(gedi.head(100))
    to_new.process()

    print("Run prediction")
    control_agbd = m.predict(to_new.train.xs)
    # within_fire['agbd_control_mean'] = control_agbd
    # within_fire['agbd_control_median'] = control_agbd
    # return within_fire
    print(control_agbd)
