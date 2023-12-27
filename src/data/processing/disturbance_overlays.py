from fastai.tabular.all import save_pickle
from src.data.adapters import disturbance_agents as da
from src.data.processing import gedi_raster_matching
from src.data.utils import gedi_utils, raster
from pathlib import Path
import rasterio as rio
from src.utils.logging_util import get_logger
import geopandas as gpd
from shapely.geometry import box
from src.constants import WGS84
import pandas as pd
import importlib
importlib.reload(raster)

logger = get_logger(__file__)

RASTER_BANDS = [f"year_{year}" for year in range(1985, 2022)]


def overlay_with_disturbances(input_path: str, output_path: str):
    gedi_to_be_matched = gedi_utils.get_gedi_shots(input_path)
    gedi_matched = []

    logger.info("Starting raster mathching")
    raster_files = list(Path(da.DATASET_PATH).iterdir())
    for file_name in raster_files:
        raster_bounds = box(*rio.open(file_name).bounds)
        bounds_gdf = gpd.GeoDataFrame(
            index=[0],
            crs=WGS84,
            geometry=[raster_bounds])

        gedi_within = gedi_to_be_matched.sjoin(
            bounds_gdf, how="inner", predicate="within")

        if len(gedi_within) == 0:
            continue

        logger.info(f"Matching {len(gedi_within)} gedi shots.")
        gedi_to_be_matched.drop(gedi_within.index, inplace=True)

        matched = gedi_raster_matching.sample_raster(
            raster.RasterSampler(file_name, RASTER_BANDS),
            gedi_within,
            # Raster resolution is 30m, so use 2x2 matching with GEDI.
            kernel=2
        )
        filtered = filter_disturbances(matched)
        gedi_matched.append(filtered)

    overlay = pd.concat(gedi_matched)
    save_pickle(output_path, overlay)
    return overlay


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

        per_year = pd.concat([col_mean, col_std, col_median], axis=1)

        # Get rid of the ones with std == 0 and median == fill value
        filtered = per_year[
            ~((per_year.da_std == 0) & (per_year.da_median == 65533.0))]

        # Add column for the year.
        filtered['da_year'] = year

        filtered_disturbances.append(filtered)
    return pd.concat(filtered_disturbances)
