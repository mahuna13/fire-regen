import pandas as pd
import rasterio as rio
from src.data.utils.raster import RasterSampler, reproject_raster
from src.utils.logging_util import get_logger

logger = get_logger(__file__)


class Raster:
    def __init__(self, raster_file_path: str, bands: dict[str, int]):
        self.raster = self.read_raster(raster_file_path)
        self.bands = bands

    def read_raster(self, directory: str) -> rio.DatasetReader:
        '''
        Open raster as array file and evaluate if is in right projection
        compared with GEDI data (ESPG:4326).
        '''
        raster = rio.open(directory)
        crs = str(raster.crs)
        if crs == 'EPSG:4326':
            return raster
        else:
            raise Warning("Your raster file is not in crs 'EPSG:4326'")

    def get_band_index(self, band_name: str) -> int:
        return self.bands[band_name]

    def transform_geo_to_xy_coords(self, geo_coords: list[tuple]):
        transformer = rio.transform.AffineTransformer(self.raster.transform)
        return [transformer.rowcol(x[0], x[1]) for x in geo_coords]

    def transform_xy_to_geo_coords(self, xy_coords: list[tuple]):
        transformer = rio.transform.AffineTransformer(self.raster.transform)
        return [transformer.xy(x[0], x[1]) for x in xy_coords]

    def sample(self, xy_coords: list[tuple]):
        return self.raster.sample(xy_coords)


class FireRastersDB:
    RASTER_TYPES = ['dnbr', 'dnbr6', 'rdnbr']
    bands = {'severity': 0}

    def __init__(self, raster_file_path: str, fire_name: str):
        self.file_path = raster_file_path
        self.fire_name = fire_name

        self.rasters = {}
        for raster_type in self.RASTER_TYPES:
            self.rasters[raster_type] = Raster(
                self._get_tif_file_path(raster_type), self.bands)

    def _get_tif_file_path(self, raster_type):
        return f'{self.file_path}{self.fire_name}_{raster_type}.tif'

    def get(self, raster_type):
        return self.rasters[raster_type]


def match_gedi_to_raster(gedi_shots: pd.DataFrame, raster: Raster,
                         kernel_size: int, bands: list[str]):
    raster_sampler = RasterSampler(raster)
    return raster_sampler.sample(gedi_shots,
                                 'lon_lowestmode',
                                 'lat_lowestmode',
                                 kernel_size,
                                 bands)


def reproject_rasters(directory: str, fire_id: str, fire_name: str):
    RASTER_TYPES = ['dnbr', 'dnbr6', 'rdnbr']
    for raster_type in RASTER_TYPES:
        raster_old = f'{directory}{fire_id}_{raster_type}.tif'
        raster_new = f'{directory}{fire_name}_{raster_type}.tif'
        reproject_raster(raster_old, raster_new)
