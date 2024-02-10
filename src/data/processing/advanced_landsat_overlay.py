import importlib
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio as rio
from shapely.geometry import box
from src.constants import DATA_PATH, WGS84
from src.data.processing import gedi_raster_matching, overlay
from src.data.utils import gedi_utils, raster
from src.utils.logging_util import get_logger

importlib.reload(raster)

logger = get_logger(__file__)


def ADV_LANDSAT_PATH(year):
    return f"{DATA_PATH}/rasters/LANDSAT_ADVANCED/{year}"


BANDS = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "NDVI",
         "NDWI", "NBR", "NDMI", "SWIRS", "SVVI", "brightness", "greenness",
         "wetness"]


def MONTHLY_LANDSAT_PATH(year):
    return f"{DATA_PATH}/rasters/LANDSAT_MONTHLY/{year}"


def overlay_advanced_landsat(
        df: pd.DataFrame,
        year: int,
        kind: str):
    df = overlay.validate_input(df)
    gdf = gedi_utils.convert_to_geo_df(df)
    bands = [f"{band}_{kind}" for band in BANDS]
    gedi_matched = []

    logger.info("Starting raster matching.")
    raster_files = list(Path(f"{ADV_LANDSAT_PATH(year)}/{kind}").iterdir())
    for file_name in raster_files:
        logger.info(f"Processing file name: {file_name} \n \n")
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
            raster.RasterSampler(file_name, bands),
            gedi_within,
            # Raster resolution is 30m, so use 2x2 matching with GEDI.
            kernel=2
        )

        for column in bands:
            gedi_within[column] = matched[f"{column}_mean"]
        gedi_matched.append(gedi_within)

    return pd.concat(gedi_matched)


def overlay_monthly_landsat(
        df: pd.DataFrame,
        year: int,
        month: int):
    df = overlay.validate_input(df)
    gdf = gedi_utils.convert_to_geo_df(df)
    monthly_bands = gedi_raster_matching.get_landsat_bands(year)
    bands = [f"{band}_{month}" for band in monthly_bands]
    gedi_matched = []

    logger.info("Starting raster matching.")
    all_files = list(Path(f"{MONTHLY_LANDSAT_PATH(year)}").iterdir())
    monthly_rasters = [filename for filename in all_files
                       if f"{year}_{month}-" in filename.name]
    for file_name in monthly_rasters:
        logger.info(f"Processing file name: {file_name} \n \n")
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
            raster.RasterSampler(file_name, bands),
            gedi_within,
            # Raster resolution is 30m, so use 2x2 matching with GEDI.
            kernel=2
        )

        for column in bands:
            gedi_within[column] = matched[f"{column}_mean"]
        gedi_matched.append(gedi_within)

    if len(gedi_matched) == 0:
        logger.info(f"No matches found for year: {year} and month: {month}")
        return None

    return pd.concat(gedi_matched)
