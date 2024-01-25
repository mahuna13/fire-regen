import rasterio
from rasterio.merge import merge
import os
from src.utils.logging_util import get_logger
from src.constants import DATA_PATH

logger = get_logger(__file__)
LANDSAT_BANDS = ['nbr', 'ndvi', 'SR_B1', 'SR_B2',
                 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
TERRAIN_OUTPUT_DIR = f"{DATA_PATH}/rasters/TERRAIN/"
TERRAIN_PREFIX = "terrain_"
TERRAIN_BANDS = ['elevation', 'slope', 'aspect', 'soil']


def INPUT_DIR(year): return f"{DATA_PATH}/rasters/LANDSAT/{year}/"
def OUTPUT_DIR(
    year): return f"{DATA_PATH}/rasters/LANDSAT/{year}/out/"


def PREFIX(year): return f"landsat_{year}_"


def merge_partial_rasters(year, bands):
    for band in bands:
        print(band)
        file_prefix = PREFIX(year) + band
        band_tifs = [filename for filename in os.listdir(
            INPUT_DIR(year)) if filename.startswith(file_prefix)]

        print(file_prefix)
        srcs_to_mosaic = []
        for band_tif in band_tifs:
            src = rasterio.open(INPUT_DIR(year) + band_tif)
            srcs_to_mosaic.append(src)

        mosaic, out_trans = merge(srcs_to_mosaic)

        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                        "height": mosaic.shape[1],
                         "width": mosaic.shape[2],
                         "transform": out_trans})
        with rasterio.open(f"{OUTPUT_DIR(year)}{PREFIX(year)}{band}.tif",
                           "w",
                           **out_meta) as dest:
            dest.write(mosaic)


def merge_bands(dir_path, prefix, bands):
    file_list = [f"{dir_path}{prefix}{band}.tif" for band in bands]

    # Read metadata of first file
    with rasterio.open(file_list[0]) as src0:
        meta = src0.meta

    # Update meta to reflect the number of layers
    meta.update(count=len(file_list))

    # Read each layer and write it to stack
    stack_filename = f'{dir_path}{prefix}stack.tif'
    logger.debug(f"Writing stack file - {stack_filename}")
    with rasterio.open(stack_filename, 'w', **meta) as dst:
        for id, layer in enumerate(file_list, start=1):
            with rasterio.open(layer) as src1:
                dst.write_band(id, src1.read(1))


def process_landsat_rasters(start_year, end_year):
    for year in range(start_year, end_year):
        logger.debug(f"Processing LANDSAT rasters for year {year}")

        merge_partial_rasters(year, LANDSAT_BANDS)
        merge_bands(OUTPUT_DIR(year), PREFIX(year), LANDSAT_BANDS)


def process_terrain_rasters():
    logger.debug("Processing TERRAIN rasters")

    merge_bands(TERRAIN_OUTPUT_DIR, TERRAIN_PREFIX, TERRAIN_BANDS)
