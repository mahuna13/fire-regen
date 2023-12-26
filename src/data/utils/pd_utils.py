from typing import Callable
import geopandas as gpd
import pandas as pd
from src.utils.logging_util import get_logger

logger = get_logger(__file__)


def read_chunks_as_gdf(get_file_name: Callable[[int], str],
                       num_of_chunks: int) -> gpd.GeoDataFrame:
    gpds = []
    for i in range(num_of_chunks):
        logger.info(f'Processing chunk {i}')
        chunk = pd.read_csv(get_file_name(i), index_col=0)
        gpds.append(chunk)

    logger.debug('Joining dataframes')
    return gpd.GeoDataFrame(pd.concat(gpds, ignore_index=True))
