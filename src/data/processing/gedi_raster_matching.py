from pathlib import Path

import geopandas as gpd
import pandas as pd
from src.constants import DATA_PATH
from src.data.utils import raster
from src.utils.logging_util import get_logger

logger = get_logger(__file__)

BURN_DATA_RASTER = f"{DATA_PATH}/rasters/burn_data_sierras.tif"
LAND_COVER_RASTER = f"{DATA_PATH}/rasters/land_cover_sierras.tif"
TERRAIN_RASTER = f"{DATA_PATH}/rasters/TERRAIN/terrain_stack.tif"


def LANDSAT_RASTER(year):
    landsat_num = which_landsat(year)
    file_name = f"landsat{landsat_num}_{year}.tif"
    return f"{DATA_PATH}/rasters/LANDSAT/{year}/{file_name}"


def DYNAMIC_WORLD_RASTER(year):
    return f"{DATA_PATH}/rasters/DYNAMIC_WORLD/dynamic_world_{year}.tif"


def LCSM_RASTER(year):
    return f"{DATA_PATH}/rasters/lcsm/lcms_{year}.tif"


BURN_RASTER_BANDS = ['burn_severity', 'burn_year', 'burn_counts']
LAND_COVER_BANDS = ['land_cover']
TERRAIN_BANDS = ['aspect', 'elevation', 'slope', 'soil']
LANDSAT5_BANDS = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'NDVI']
LANDSAT8_BANDS = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6",
                  "SR_B7", "NDVI"]
DW_BANDS = ["dw_land_cover"]


def get_landsat_raster_sampler(year):
    raster_path = LANDSAT_RASTER(year)
    bands = get_landsat_bands(year)
    return raster.RasterSampler(raster_path, bands)


def get_ndvi_raster_sampler(year):
    raster_path = LANDSAT_RASTER(year)
    bands_map = get_ndvi_band(year)
    return raster.RasterSampler(raster_path, bands_map=bands_map)


def get_dw_raster_sampler(year):
    raster_path = DYNAMIC_WORLD_RASTER(year)
    return raster.RasterSampler(raster_path, DW_BANDS)


def which_landsat(year):
    if year < 1999:
        return 5
    elif year < 2013:
        return 7
    else:
        return 8


def get_landsat_bands(year):
    landsat_num = which_landsat(year)
    if landsat_num == 5 or landsat_num == 7:
        return LANDSAT5_BANDS
    else:
        return LANDSAT8_BANDS


def get_ndvi_band(year):
    landsat_num = which_landsat(year)
    if landsat_num == 5 or landsat_num == 7:
        return {6: "ndvi"}
    else:
        return {7: "ndvi"}


def match_burn_raster(
    gedi: gpd.GeoDataFrame,
    kernel: int
) -> gpd.GeoDataFrame:
    burn_raster = raster.RasterSampler(BURN_DATA_RASTER, BURN_RASTER_BANDS)

    return sample_raster(burn_raster, gedi, kernel)


def match_landcover(
    year: int,
    df: gpd.GeoDataFrame,
    kernel: int
) -> gpd.GeoDataFrame:
    lcsm_raster = raster.RasterSampler(
        LCSM_RASTER(year), LAND_COVER_BANDS)

    return sample_raster(lcsm_raster, df, kernel)


def match_burn_landcover(
    gedi: gpd.GeoDataFrame,
    kernel: int
) -> gpd.GeoDataFrame:
    matches = []
    for burn_year in gedi.burn_year.astype('int').unique():
        logger.debug(f"Matching land cover for year {burn_year}.")

        gedi_year = gedi[gedi.burn_year == burn_year]
        match = match_landcover_for_year(burn_year, gedi_year, kernel)
        matches.append(match)
    return pd.concat(matches)


def match_landcover_for_year(
    year: int,
    df: gpd.GeoDataFrame,
    kernel: int
) -> gpd.GeoDataFrame:
    if year < 1986:
        raster_year = 1985
    else:
        raster_year = year - 1

    return match_landcover(raster_year, df, kernel)


def match_terrain(
    df: gpd.GeoDataFrame,
    kernel: int
) -> gpd.GeoDataFrame:
    terrain_raster = raster.RasterSampler(TERRAIN_RASTER, TERRAIN_BANDS)

    return sample_raster(terrain_raster, df, kernel)


def sample_raster(
    raster_sampler: raster.RasterSampler,
    df: gpd.GeoDataFrame,
    kernel: int,
    expanded: bool = False
):
    LAT = 'latitude'
    LON = 'longitude'
    if kernel == 1:
        return raster_sampler.sample(df, LON, LAT)
    elif kernel == 2:
        return raster_sampler.sample_2x2(df, LON, LAT, expanded=expanded)
    elif kernel == 3:
        return raster_sampler.sample_3x3(df, LON, LAT)


def merge_landsat_tiles_for_year(year):
    logger.debug(f"Merging tiles for year {year}")
    landsat = which_landsat(year)
    path = Path(f"{DATA_PATH}/rasters/LANDSAT/{year}")
    output_file_path = f"{path}/landsat{landsat}_{year}.tif"

    raster.merge_raster_tiles(path, output_file_path)


def merge_dynamic_world_tiles_for_year(year):
    logger.debug(f"Merging tiles for year {year}")
    path = Path(f"{DATA_PATH}/rasters/DYNAMIC_WORLD/{year}")
    output_file_path = \
        f"{DATA_PATH}/rasters/DYNAMIC_WORLD/dynamic_world_{year}.tif"

    raster.merge_raster_tiles(path, output_file_path)
