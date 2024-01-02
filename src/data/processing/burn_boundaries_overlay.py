# This module identifies all GEDI shots that fall within a X meters boundary of
# any previous fire perimeter.

# This will be used to filter out those shots, as the GEDI geo location error
# makes it uncertain if the shot fell inside or outside the fire perimeter.

# We use CalFire perimeters that include small fires, with large historic
# record.

import pandas as pd
from fastai.tabular.all import load_pickle, save_pickle
from src.data.adapters import mtbs
from src.data.utils import gedi_utils
from src.utils.logging_util import get_logger

logger = get_logger(__file__)


def overlay_with_boundary_buffers(
        input_path: str,
        output_path: str,
        distance: int = 30):
    gedi_shots = gedi_utils.get_gedi_shots(input_path)

    buffers = load_pickle(mtbs.MTBS_BOUNDARY_BUFFER(distance))

    intersection = gedi_shots.sjoin(buffers,
                                    how="left",
                                    predicate="within")

    shots_around_boundaries = intersection[intersection.index_right.notna()]
    shots_around_boundaries["most_recent_boundary"] = \
        shots_around_boundaries.groupby(gedi_utils.INDEX).Ig_Year.max()

    # Save.
    return _save_overlay(shots_around_boundaries, output_path)


def _save_overlay(df: pd.DataFrame, output_path: str):
    df = df.drop(columns=["longitude", "latitude", "geometry"])

    logger.info("Saving the results.")
    save_pickle(output_path, df)
    return df
