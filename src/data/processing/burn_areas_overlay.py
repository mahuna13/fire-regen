# This module identifies all GEDI shots that fall within any burned area as
# identified by the Calfire historical fires dataset.

# We exclude a 10m buffer around the fire perimeter, to decrease the impact of
# GEDI's geolocation uncertainty.

import numpy as np
import pandas as pd
from fastai.tabular.all import load_pickle, save_pickle
from src.data.adapters import calfire_perimeters as cp
from src.data.utils import gedi_utils
from src.utils.logging_util import get_logger

logger = get_logger(__file__)


def overlay_with_burn_area(input_path: str, output_path: str):
    gedi_shots = gedi_utils.get_gedi_shots(input_path)

    burn_areas = load_pickle(cp.CALFIRE_BURN_AREA_TRIMMED_10m)

    intersection = gedi_shots.sjoin(burn_areas,
                                    how="left",
                                    predicate="within")

    burned_shots = intersection[intersection.index_right.notna()].astype({
        'YEAR_': 'int64'})
    recent_burned_shots = burned_shots[burned_shots.YEAR_ > 1984]

    most_recent_year = burned_shots.groupby(gedi_utils.INDEX).YEAR_.max()
    fire_count = burned_shots.groupby(gedi_utils.INDEX).YEAR_.count()
    recent_fire_count = recent_burned_shots.groupby(
        gedi_utils.INDEX).YEAR_.count()

    # Add columns.
    within_col = "burned"
    year_col = "most_recent_fire_year"
    count_col = "fire_count"
    recent_fire_count_col = "recent_fire_count"
    gedi_shots[within_col] = False
    gedi_shots[year_col] = np.nan
    gedi_shots[count_col] = 0
    gedi_shots[recent_fire_count_col] = 0

    # Populate columns.
    gedi_shots.loc[burned_shots.index, within_col] = True
    gedi_shots.loc[burned_shots.index, year_col] = most_recent_year
    gedi_shots.loc[burned_shots.index, count_col] = fire_count
    gedi_shots.loc[recent_burned_shots.index,
                   recent_fire_count_col] = recent_fire_count

    # Save.
    gedi_shots = _save_overlay(gedi_shots, output_path)
    return gedi_shots


def _save_overlay(df: pd.DataFrame, output_path: str):
    df = df.drop(columns=["longitude", "latitude", "geometry"])

    logger.info("Saving the results.")
    save_pickle(output_path, df)
    return df
