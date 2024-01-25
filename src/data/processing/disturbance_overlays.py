import importlib
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio as rio
from shapely.geometry import box
from src.constants import WGS84
from src.data.adapters import disturbance_agents as da
from src.data.processing import gedi_raster_matching, overlay
from src.data.utils import gedi_utils, raster
from src.utils.logging_util import get_logger

importlib.reload(raster)

logger = get_logger(__file__)

RASTER_BANDS = [f"year_{year}" for year in range(1985, 2022)]


def overlay_with_disturbances(df: pd.DataFrame):
    df = overlay.validate_input(df)
    gdf = gedi_utils.convert_to_geo_df(df)
    gedi_matched = []

    logger.info("Starting raster mathching")
    raster_files = list(Path(da.DATASET_PATH).iterdir())
    for file_name in raster_files:
        raster_bounds = box(*rio.open(file_name).bounds)
        bounds_gdf = gpd.GeoDataFrame(
            index=[0],
            crs=WGS84,
            geometry=[raster_bounds])

        gedi_within = gdf.sjoin(bounds_gdf, how="inner", predicate="within")

        if len(gedi_within) == 0:
            continue

        logger.info(f"Matching {len(gedi_within)} gedi shots.")
        gdf.drop(gedi_within.index, inplace=True)

        matched = gedi_raster_matching.sample_raster(
            raster.RasterSampler(file_name, RASTER_BANDS),
            gedi_within,
            # Raster resolution is 30m, so use 2x2 matching with GEDI.
            kernel=2,
            expanded=True
        )
        filtered = filter_disturbances(matched)
        gedi_matched.append(filtered)

    return pd.concat(gedi_matched)


def filter_disturbances(df: pd.DataFrame):
    filtered_disturbances = []
    for year in range(1985, 2022):
        col_mean = pd.melt(
            df,
            value_vars=[f'year_{year}_mean'],
            value_name="da_mean",
            ignore_index=False
        ).drop(columns=['variable'])

        col_std = pd.melt(
            df,
            value_vars=[f'year_{year}_std'],
            value_name="da_std",
            ignore_index=False
        ).drop(columns=['variable'])

        col_median = pd.melt(
            df,
            value_vars=[f'year_{year}_median'],
            value_name="da_median",
            ignore_index=False
        ).drop(columns=['variable'])

        col_min = pd.melt(
            df,
            value_vars=[f'year_{year}_min'],
            value_name="da_min",
            ignore_index=False
        ).drop(columns=['variable'])

        col_max = pd.melt(
            df,
            value_vars=[f'year_{year}_max'],
            value_name="da_max",
            ignore_index=False
        ).drop(columns=['variable'])

        per_year = pd.concat([col_mean, col_std, col_median,
                              col_min, col_max], axis=1)

        # Get rid of the ones with std == 0 and median == fill value
        filtered = per_year[
            ~((per_year.da_std == 0) & (per_year.da_median == 65533.0))]

        # Add column for the year.
        filtered['da_year'] = year

        filtered_disturbances.append(filtered)
    return pd.concat(filtered_disturbances)
