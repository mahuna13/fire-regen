# This module identifies all GEDI shots that fall within a X meters boundary of
# any previous fire perimeter.

# This will be used to filter out those shots, as the GEDI geo location error
# makes it uncertain if the shot fell inside or outside the fire perimeter.

# We use CalFire perimeters that include small fires, with large historic
# record.

import pandas as pd
from fastai.tabular.all import load_pickle
from src.data.adapters import mtbs
from src.data.processing import overlay
from src.data.utils import gedi_utils
from src.utils.logging_util import get_logger

logger = get_logger(__file__)


# Returns shorts within distance from a fire boundary.
def overlay_with_boundary_buffers(
        df: pd.DataFrame,
        distance: int = 30):
    df = overlay.validate_input(df)
    gdf = gedi_utils.convert_to_geo_df(df)

    buffers = load_pickle(mtbs.MTBS_BOUNDARY_BUFFER(distance))

    intersection = gdf.sjoin(buffers,
                             how="left",
                             predicate="within")

    shots_around_boundaries = intersection[intersection.index_right.notna()]
    shots_around_boundaries["most_recent_boundary"] = \
        shots_around_boundaries.groupby(overlay.INDEX).Ig_Year.max()

    shots_around_boundaries.drop(columns=["index_right"], inplace=True)

    # Save.
    return shots_around_boundaries
