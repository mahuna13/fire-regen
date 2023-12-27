from src.constants import DATA_PATH, WGS84
from src.data.utils import raster
from pathlib import Path
from src.utils.logging_util import get_logger

logger = get_logger(__file__)

DATASET_PATH = f"{DATA_PATH}/disturbance_agents"
RASTER = f"{DATASET_PATH}/disturbance_agents.tif"


# The dataset comes broken down into multiple tif files - we merge them all
# into one to do matching with GEDI. It would be better to keep them separate
# once we use Sedona for processing, but for now this is easier but definitely
# less efficient.
def merge_tiles():
    path = Path(DATASET_PATH)
    raster.merge_raster_tiles(path, RASTER)


def reproject_raster():
    raster_files = list(Path(DATASET_PATH).iterdir())
    for file_name in raster_files:
        logger.info(f"Processing file: {file_name}")
        raster.reproject_raster(file_name, file_name, dst_crs=WGS84)
