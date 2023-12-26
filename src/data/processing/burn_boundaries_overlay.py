# This module identifies all GEDI shots that fall within a 10m boundary of any
# previous fire perimeter.

# This will be used to filter out those shots, as the GEDI geo location error
# makes it uncertain if the shot fell inside or outside the fire perimeter.

# We use CalFire perimeters that include small fires, with large historic
# record.

import numpy as np
import pandas as pd
from fastai.tabular.all import load_pickle, save_pickle
from src.data.adapters import calfire_perimeters as cp
from src.data.utils import gedi_utils
from src.utils.logging_util import get_logger

logger = get_logger(__file__)


def overlay_with_boundary_buffers(input_path: str, output_path: str):
    gedi_shots = gedi_utils.get_gedi_shots(input_path)

    buffers = load_pickle(cp.CALFIRE_BOUNDARY_BUFFER_10m)

    intersection = gedi_shots.sjoin(buffers,
                                    how="left",
                                    predicate="within")

    shots_around_boundaries = intersection[intersection.index_right.notna()]
    most_recent_year = shots_around_boundaries.groupby(
        gedi_utils.INDEX).YEAR_.max()

    # Add columns.
    within = "within_fire_boundary"
    year = "most_recent_boundary"
    gedi_shots[within] = False
    gedi_shots[year] = np.nan

    # Populate columns.
    gedi_shots.loc[shots_around_boundaries.index, within] = True
    gedi_shots.loc[shots_around_boundaries.index, year] = most_recent_year

    # Save.
    gedi_shots = _save_overlay(gedi_shots, output_path)
    return gedi_shots


def _save_overlay(df: pd.DataFrame, output_path: str):
    df = df.drop(columns=["longitude", "latitude", "geometry"])

    logger.info("Saving the results.")
    save_pickle(output_path, df)
    return df
