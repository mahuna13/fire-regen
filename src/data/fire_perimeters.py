import fiona
import geopandas as gpd
import pandas as pd

from src.data import gedi_loader
from src.data.raster import RasterSampler, Raster, reproject_raster
from src.utils.logging_util import get_logger

logger = get_logger(__file__)


class FirePerimetersDB:
    '''
    FirePerimeters class can load CALFIRE fire perimeter database, and be
    queried for individual fires.

    Some of the code is specific to how CALFIRE stores its data, so it acts
    like a data adapter as well.
    '''

    def __init__(self, path_to_dataset, crs: int = 4326):
        layers = fiona.listlayers(path_to_dataset)
        '''
        CALFIRE dataset has three layers:
           1. 'firep21_2' is fire perimeters.
           2. 'rxburn21_2' are prescribed burns.
           3. 'Non_RXFire_Legacy13_2' seem to be other treatment types.
         We want fire perimeters here.
        '''
        gdf_fire_perimeters = gpd.read_file(path_to_dataset, layer=layers[0])
        self.perimeters = gdf_fire_perimeters.to_crs(crs)

    def get_perimeters_gdf(self):
        return self.perimeters


class FirePerimeters:
    def __init__(self, db_handle: FirePerimetersDB):
        self.perimeters = db_handle.get_perimeters_gdf()

    def filter_for_years(self, years: list[str]):
        ''' Filter fire perimeters to include only the years requested. '''
        self.perimeters = self.perimeters[self.perimeters.YEAR_.isin(years)]
        return self

    def filter_within_geometry(self, query_gpd: gpd.GeoDataFrame):
        ''' Filter fire perimeters to be within geometries provided. '''
        self.perimeters = self.perimeters.sjoin(
            query_gpd, how="inner", predicate="within")

        # Remove the index that was added in the sjoin, since it's not needed.
        self.perimeters.drop(columns="index_right", inplace=True)
        return self

    def count(self):
        return self.perimeters.shape[0]

    def get_largest_fires(self, count: int = 10):
        return self.perimeters.sort_values('Shape_Area', ascending=False).head(count).FIRE_NAME

    def get_fire(self, fire_name: str):
        return Fire(self.perimeters[self.perimeters.FIRE_NAME == fire_name])


class Fire:
    def __init__(self, calfire_fire_info: gpd.GeoDataFrame):
        self.fire = calfire_fire_info
        self.alarm_date = pd.to_datetime(self.fire.ALARM_DATE.iloc[0])
        self.cont_date = pd.to_datetime(self.fire.CONT_DATE.iloc[0])

        # Create a gdf to store the area around the fire, called fire buffer.
        fire_geometry = self.fire.geometry.iloc[0]
        self.fire_buffer = gpd.GeoSeries([fire_geometry.buffer(
            1000).symmetric_difference(fire_geometry)])

    def get_buffer(self, width: int, exclusion_zone: int = 100):
        # convert to projected CRS to be able to specify distances in meters.
        fire_projected = self.fire.to_crs(epsg=3310)

        fire_geom = fire_projected.geometry.iloc[0]

        exclude = fire_geom.buffer(
            exclusion_zone).union(fire_geom)
        buffer = exclude.buffer(width).symmetric_difference(exclude)

        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([buffer]), crs=3310) \
            .to_crs(self.fire.crs)

    def overlay_fire_map(self, gdf: gpd.GeoDataFrame):
        self.fire.overlay(gdf, how="union").plot(cmap='tab20b')

    def load_gedi(self, load_buffer: bool = False):
        ''' Loads GEDI shots stored for the fire from postgress database. '''
        self.gedi = gedi_loader.get_gedi_shots(self.fire.geometry)

        if load_buffer:
            self.gedi_buffer = gedi_loader.get_gedi_shots(
                self.fire_buffer.geometry)

    def get(self):
        return self.fire

    def get_gedi_before_fire(self):
        return self.gedi[self.gedi.absolute_time < self.alarm_date]

    def get_gedi_buffer_before_fire(self):
        return self.gedi_buffer[
            self.gedi_buffer.absolute_time < self.alarm_date]

    def get_gedi_after_fire(self):
        return self.gedi[self.gedi.absolute_time > self.cont_date]

    def get_gedi_buffer_after_fire(self):
        return self.gedi_buffer[
            self.gedi_buffer.absolute_time > self.cont_date]


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
