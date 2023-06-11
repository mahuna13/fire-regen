import geopandas as gpd
import pandas as pd
from src.data import raster
from src.utils.logging_util import get_logger

logger = get_logger(__file__)

BURN_DATA_RASTER = '/maps/fire-regen/data/rasters/burn_data_sierras.tif'
LAND_COVER_RASTER = '/maps/fire-regen/data/rasters/land_cover_sierras.tif'
TERRAIN_RASTER = '/maps/fire-regen/data/rasters/TERRAIN/terrain_stack.tif'


def LANDSAT_RASTER(year):
    return f"/maps/fire-regen/data/rasters/LANDSAT/{year}/out/landsat_{year}_stack.tif"


def DYNAMIC_WORLD_RASTER(year):
    return f"/maps/fire-regen/data/rasters/DYNAMIC_WORLD/dynamic_world_{year}.tif"


def LCSM_RASTER(year):
    return f"/maps/fire-regen/data/rasters/lcsm/lcms_{year}.tif"


BURN_RASTER_BANDS = {0: 'burn_severity', 1: 'burn_year', 2: 'burn_counts'}
LAND_COVER_BANDS = {0: 'land_cover'}
TERRAIN_BANDS = {0: 'elevation', 1: 'slope', 2: 'aspect', 3: 'soil'}
LANDSAT_BANDS = {0: 'nbr', 1: 'ndvi', 2: 'SR_B1', 3: 'SR_B2',
                 4: 'SR_B3', 5: 'SR_B4', 6: 'SR_B5', 7: 'SR_B6', 8: 'SR_B7'}


def match_burn_raster(
    gedi: gpd.GeoDataFrame,
    kernel: int
) -> gpd.GeoDataFrame:
    burn_raster = raster.RasterSampler(BURN_DATA_RASTER, BURN_RASTER_BANDS)

    if kernel == 1:
        return burn_raster.sample(gedi, 'longitude', 'latitude')
    elif kernel == 2:
        return burn_raster.sample_2x2(gedi, 'longitude', 'latitude')
    elif kernel == 3:
        return burn_raster.sample_3x3(gedi, 'longitude', 'latitude')


def match_burn_landcover(
    gedi: gpd.GeoDataFrame,
    kernel: int
) -> gpd.GeoDataFrame:
    matches = []
    for burn_year in gedi.burn_year_median.astype('int').unique():
        logger.debug(f"Matching land cover for year {burn_year}.")

        gedi_year = gedi[gedi.burn_year_median == burn_year]
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

    lcsm_raster = raster.RasterSampler(
        LCSM_RASTER(raster_year), LAND_COVER_BANDS)

    if kernel == 1:
        return lcsm_raster.sample(df, 'longitude', 'latitude')
    elif kernel == 2:
        return lcsm_raster.sample_2x2(df, 'longitude', 'latitude')
    elif kernel == 3:
        return lcsm_raster.sample_3x3(df, 'longitude', 'latitude')
