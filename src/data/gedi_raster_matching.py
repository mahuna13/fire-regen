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

BURN_DATA_RASTER = '/maps/fire-regen/data/rasters/burn_data_sierras.tif'
LAND_COVER_RASTER = '/maps/fire-regen/data/rasters/land_cover_sierras.tif'
TERRAIN_RASTER = '/maps/fire-regen/data/rasters/TERRAIN/terrain_stack.tif'


def LANDSAT_RASTER(year):
    return f"/maps/fire-regen/data/rasters/LANDSAT/{year}/out/landsat_{year}_stack.tif"


def LANDSAT5_RASTER(year):
    return f"/maps/fire-regen/data/rasters/LANDSAT/LANDSAT5/{year}/landsat5_{year}.tif"


def LANDSAT8_ADV_RASTER(year):
    return f"/maps/fire-regen/data/rasters/LANDSAT/{year}/landsat8_{year}.tif"


def DYNAMIC_WORLD_RASTER(year):
    return f"/maps/fire-regen/data/rasters/DYNAMIC_WORLD/dynamic_world_{year}.tif"


def LCSM_RASTER(year):
    return f"/maps/fire-regen/data/rasters/lcsm/lcms_{year}.tif"


BURN_RASTER_BANDS = {0: 'burn_severity', 1: 'burn_year', 2: 'burn_counts'}
LAND_COVER_BANDS = {0: 'land_cover'}
TERRAIN_BANDS = {0: 'elevation', 1: 'slope', 2: 'aspect', 3: 'soil'}
LANDSAT_BANDS = {0: 'nbr', 1: 'ndvi', 2: 'SR_B1', 3: 'SR_B2',
                 4: 'SR_B3', 5: 'SR_B4', 6: 'SR_B5', 7: 'SR_B6', 8: 'SR_B7'}
LANDSAT5_BANDS = {0: 'SR_B1', 1: 'SR_B2',
                  2: 'SR_B3', 3: 'SR_B4', 4: 'SR_B5', 5: 'SR_B7',
                  6: 'ndvi'}
LANDSAT8_ADV_BANDS = {0: "SR_B1", 1: "SR_B2", 2: "SR_B3", 3: "SR_B4",
                      4: "SR_B5", 5: "SR_B6", 6: "SR_B7", 7: "NDVI", 8: "NDWI",
                      9: "NBR", 10: "NDMI", 11: "SWIRS", 12: "SVVI",
                      13: "brightness", 14: "greenness", 15: "wetness"}


def get_landsat_raster_sampler(year):
    if year < 1999:
        return raster.RasterSampler(LANDSAT5_RASTER(year), LANDSAT5_BANDS)
    elif year < 2018:
        return raster.RasterSampler(LANDSAT8_ADV_RASTER(year), LANDSAT8_ADV_BANDS)
    else:
        return raster.RasterSampler(LANDSAT_RASTER(year), LANDSAT_BANDS)


def which_landsat(year):
    if year < 1999:
        return 5
    elif year < 2013:
        return 7
    else:
        return 8


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
