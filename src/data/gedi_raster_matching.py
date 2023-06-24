import geopandas as gpd
import pandas as pd
from src.data import raster
import os
from pathlib import Path
from src.constants import DATA_PATH
import rasterio as rio
from rasterio.merge import merge
from src.utils.logging_util import get_logger

logger = get_logger(__file__)

BURN_DATA_RASTER = f"{DATA_PATH}/rasters/burn_data_sierras.tif"
LAND_COVER_RASTER = f"{DATA_PATH}/rasters/land_cover_sierras.tif"
TERRAIN_RASTER = f"{DATA_PATH}/rasters/TERRAIN/terrain_stack.tif"


def LANDSAT_RASTER(year):
    landsat_num = which_landsat(year)
    return f"{DATA_PATH}/rasters/LANDSAT/{year}/landsat{landsat_num}_{year}.tif"


def LANDSAT5_RASTER(year):
    return f"{DATA_PATH}/rasters/LANDSAT/LANDSAT5/{year}/landsat5_{year}.tif"


def LANDSAT8_ADV_RASTER(year):
    return f"{DATA_PATH}/rasters/LANDSAT/{year}/landsat8_{year}.tif"


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
LANDSAT8_ADV_BANDS = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6",
                      "SR_B7", "NDVI", "NDWI", "NBR", "NDMI", "SWIRS", "SVVI",
                      "brightness", "greenness", "wetness"]


def get_landsat_raster_sampler(year):
    raster_path = LANDSAT_RASTER(year)
    bands = get_landsat_bands(year)
    return raster.RasterSampler(raster_path, bands)


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


def match_burn_raster(
    gedi: gpd.GeoDataFrame,
    kernel: int
) -> gpd.GeoDataFrame:
    burn_raster = raster.RasterSampler(BURN_DATA_RASTER, BURN_RASTER_BANDS)

    return sample_raster(burn_raster, gedi, kernel)


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

    lcsm_raster = raster.RasterSampler(
        LCSM_RASTER(raster_year), LAND_COVER_BANDS)

    return sample_raster(lcsm_raster, df, kernel)


def match_terrain(
    df: gpd.GeoDataFrame,
    kernel: int
) -> gpd.GeoDataFrame:
    terrain_raster = raster.RasterSampler(TERRAIN_RASTER, TERRAIN_BANDS)

    return sample_raster(terrain_raster, df, kernel)


def sample_raster(
    raster_sampler: raster.RasterSampler,
    df: gpd.GeoDataFrame,
    kernel: int
):
    if kernel == 1:
        return raster_sampler.sample(df, 'longitude', 'latitude')
    elif kernel == 2:
        return raster_sampler.sample_2x2(df, 'longitude', 'latitude')
    elif kernel == 3:
        return raster_sampler.sample_3x3(df, 'longitude', 'latitude')


def merge_landsat_tiles_for_year(year):
    logger.debug(f"Merging tiles for year {year}")
    landsat = which_landsat(year)
    path = Path(f"{DATA_PATH}/rasters/LANDSAT/{year}")
    output_file_path = f"{path}/landsat{landsat}_{year}.tif"

    if os.path.exists(output_file_path):
        # We've merged the tiles already, early exit.
        return

    raster_files = list(path.iterdir())

    logger.debug('Load tif tiles')
    raster_to_mosaic = []
    for tif in raster_files:
        raster = rio.open(tif)
        raster_to_mosaic.append(raster)

    logger.debug('Merge rasters.')
    mosaic, output = merge(raster_to_mosaic)

    logger.debug('Write output')
    output_meta = raster.meta.copy()
    output_meta.update(
        {"driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": output,
         }
    )
    with rio.open(output_file_path, "w", **output_meta) as m:
        m.write(mosaic)
