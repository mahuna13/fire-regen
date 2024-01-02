# Module for fetching and processing calfire perimeter dataset.

# TODO: Add link for data location.

import fiona
import geopandas as gpd
import pandas as pd
from fastai.tabular.all import save_pickle
from src.constants import DATA_PATH, USER_PATH, SEKI_HULL, SIERRAS_HULL

# Fetch simplified regions of interest.
SEKI = gpd.read_file(SEKI_HULL)
SIERRAS = gpd.read_file(SIERRAS_HULL)


def CALFIRE_BURN_AREA_AUGMENTED(distance):
    return f"{DATA_PATH}/calfire/perimeters_{distance}m_augmented.pkl"


CALFIRE_PERIMETERS_PATH = f"{USER_PATH}/data/fire_perimeters.gdb/"


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

    def filter_for_region(self, region: gpd.GeoDataFrame):
        ''' Filter fire perimeters to intersect region provided. '''
        self.perimeters = self.perimeters.sjoin(
            region, how="inner", predicate="intersects")

        # Remove the index that was added in the sjoin, since it's not needed.
        self.perimeters.drop(columns="index_right", inplace=True)
        return self

    def filter_within_geometry(self, query_gpd: gpd.GeoDataFrame):
        ''' Filter fire perimeters to be within geometries provided. '''
        self.perimeters = self.perimeters.sjoin(
            query_gpd, how="inner", predicate="within")

        # Remove the index that was added in the sjoin, since it's not needed.
        self.perimeters.drop(columns="index_right", inplace=True)
        return self

    def filter_for_fires_over_1000_acres(self):
        # Shape area is in meter squared, so we need to convert to acres.
        self.perimeters["shape_area_acres"] = \
            self.perimeters.Shape_Area / 4046.8564224

        self.perimeters = \
            self.perimeters[self.perimeters.shape_area_acres > 1000]

    def count(self):
        return self.perimeters.shape[0]

    def get_largest_fires(self, count: int = 10):
        return self.perimeters.sort_values(
            'Shape_Area', ascending=False).head(count).FIRE_NAME

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
        # TODO: remove
        # self.gedi = gedi_loader.get_gedi_shots(self.fire.geometry)
        # if load_buffer:
        #    self.gedi_buffer = gedi_loader.get_gedi_shots(
        #        self.fire_buffer.geometry)
        pass

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


def get_fire_perimeters_for_sierras():
    calfire_db = FirePerimetersDB(CALFIRE_PERIMETERS_PATH)
    perimeters = FirePerimeters(
        calfire_db).filter_for_region(SIERRAS).perimeters

    # Convert area to acres.
    perimeters["Shape_Area_Acres"] = perimeters.Shape_Area / 4046.8564224
    return perimeters


def augment_fire_area(
    distance: int,
    save: bool = True
) -> gpd.GeoDataFrame:
    sierra_perimeters = get_fire_perimeters_for_sierras()
    original_crs = sierra_perimeters.crs

    # Convert to a projected CRS.
    perimeters_projected = sierra_perimeters.to_crs(epsg=3310)

    # Extract the buffers around fire boundary.
    boundary_buffers = perimeters_projected.boundary.buffer(distance)

    # Trim area to exclude the area around fire perimeter.
    augmented = perimeters_projected.union(boundary_buffers)
    augmented_gdf = gpd.GeoDataFrame(
        perimeters_projected, geometry=augmented).to_crs(original_crs)

    if save:
        save_pickle(CALFIRE_BURN_AREA_AUGMENTED(distance), augmented_gdf)

    return augmented_gdf
