# This module identifies all GEDI shots that fall within any burned area as
# identified by the Calfire historical fires dataset.

# We exclude a 10m buffer around the fire perimeter, to decrease the impact of
# GEDI's geolocation uncertainty.

import pandas as pd
from fastai.tabular.all import load_pickle
from src.data.adapters import calfire_perimeters as cp
from src.data.processing import overlay
from src.data.utils import gedi_utils
from src.utils.logging_util import get_logger

logger = get_logger(__file__)


def overlay_with_all_fires(
        df: pd.DataFrame,
        distance: int = 30):
    df = overlay.validate_input(df)
    gdf = gedi_utils.convert_to_geo_df(df)

    burn_areas = load_pickle(cp.CALFIRE_BURN_AREA_AUGMENTED(distance))

    intersection = gdf.sjoin(burn_areas,
                             how="left",
                             predicate="within")

    burned_shots = intersection[intersection.index_right.notna()].astype({
        'YEAR_': 'int64'})

    # Look only at gedi shots post fire, not pre fire (relevant for the
    # most recent fires 2019-2022 that overlap with the dates GEDI was
    # sampled at).
    delta_time = burned_shots.absolute_time.dt.year - burned_shots.YEAR_
    burned_shots["years_since_fire"] = delta_time
    burned_shots = burned_shots[burned_shots.years_since_fire >= 0]

    recent_burned_shots = burned_shots[burned_shots.YEAR_ > 1984]

    # Assign total number of all fires for each GEDI shot.
    fire_count_col = "fire_count"
    burned_shots[fire_count_col] = burned_shots.groupby(
        gedi_utils.INDEX).index_right.count()

    recent_fire_count_col = "recent_fire_count"
    burned_shots[recent_fire_count_col] = 0
    recent_fire_count = recent_burned_shots.groupby(
        gedi_utils.INDEX).YEAR_.count()
    burned_shots.loc[recent_burned_shots.index,
                     recent_fire_count_col] = recent_fire_count

    burned_shots.drop(columns=["index_right"], inplace=True)

    # Save.
    return burned_shots
