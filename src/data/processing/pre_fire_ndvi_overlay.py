# This module is only applicable for shots that burned, and for which we know
# the year of of the fire.

import pandas as pd
from src.data.processing import gedi_raster_matching, overlay
from src.utils.logging_util import get_logger

logger = get_logger(__file__)


def overlay_pre_fire_NDVI(df: pd.DataFrame):
    df = overlay.validate_input(df)

    START = 1985
    END = 2022

    for year in range(START, END):
        logger.info(f"Matching with pre-fire NDVI for year {year}.")
        ndvi_year = year - 1
        filtered = df[df.fire_ig_date.dt.year == year]

        raster = gedi_raster_matching.get_ndvi_raster_sampler(ndvi_year)
        matched = gedi_raster_matching.sample_raster(raster, filtered, 2) \
            .rename(columns={"ndvi_median": "pre_fire_ndvi"})

        df.loc[matched.index, "pre_fire_ndvi"] = matched.pre_fire_ndvi
    return df
