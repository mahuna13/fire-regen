import geopandas as gpd
import pandas as pd
from fastai.tabular.all import save_pickle
from src.constants import DATA_PATH, SIERRAS_HULL, SEKI_HULL

# Fetch simplified regions of interest.
SEKI = gpd.read_file(SEKI_HULL)
SIERRAS = gpd.read_file(SIERRAS_HULL)


def MTBS_BOUNDARY_BUFFER(distance):
    return f"{DATA_PATH}/mtbs/perimeters_{distance}m_buffers.pkl"


def MTBS_PERIMETERS_TRIMMED(distance):
    return f"{DATA_PATH}/mtbs/perimeters_{distance}m_trimmed.pkl"


MTBS_INDIVIDUAL_FIRES = f"{DATA_PATH}/mtbs/all_fires/mtbs"


class MTBSFirePerimetersDB:
    def __init__(self, region: gpd.GeoDataFrame, crs=4326):
        # MTBS dataset is in 4269 CRS.
        perimeters = gpd.read_file(f"{DATA_PATH}/mtbs/mtbs_perims_DD.shp")

        # Drop CLAREMONT fire, because it's the exact duplicate of North
        # Complex fire.
        perimeters = perimeters[perimeters.Event_ID != 'CA3985812091220200817']

        # Convert region to the same geometry.
        region_4269 = region.to_crs(4269)

        # Find fires that fall within the region.
        fires_within_region = perimeters.sjoin(
            region_4269, how="inner", predicate="intersects")
        fires_within_region.drop(columns="index_right", inplace=True)

        # Convert to a desired crs
        self.perimeters = fires_within_region.to_crs(crs)
        self.perimeters["Ig_Date"] = pd.to_datetime(self.perimeters.Ig_Date)


class MTBSFire:
    def __init__(self, fire_info: gpd.GeoDataFrame):
        self.fire = fire_info

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
        ''' Loads GEDI shots stored for the fire from postgress database. '''
        # self.gedi = gedi_loader.get_gedi_shots(self.fire.geometry)

        # if load_buffer:
        #    self.gedi_buffer = gedi_loader.get_gedi_shots(
        #        self.fire_buffer.geometry)

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


def get_mtbs_perimeters_for_sierras():
    mtbs_db = MTBSFirePerimetersDB(SIERRAS)

    columns = [
        'Event_ID',
        'BurnBndAc',
        'Incid_Name',
        'Ig_Date',
        'dNBR_offst',
        'dNBR_stdDv',
        'Low_T',
        'Mod_T',
        'High_T',
        'geometry'
    ]

    mtbs_fires = mtbs_db.perimeters[
        mtbs_db.perimeters.Incid_Type == "Wildfire"][columns]

    mtbs_fires['Ig_Year'] = mtbs_fires.Ig_Date.dt.year
    mtbs_fires['Low_T_adj'] = mtbs_fires.Low_T - mtbs_fires.dNBR_offst
    mtbs_fires['Mod_T_adj'] = mtbs_fires.Mod_T - mtbs_fires.dNBR_offst
    mtbs_fires['High_T_adj'] = mtbs_fires.High_T - mtbs_fires.dNBR_offst

    return mtbs_fires


def extract_buffers_around_perimeters(
    distance: int,
    save: bool = True
) -> gpd.GeoDataFrame:
    sierra_perimeters = get_mtbs_perimeters_for_sierras()
    original_crs = sierra_perimeters.crs

    # Convert to a projected CRS.
    perimeters_projected = sierra_perimeters.to_crs(epsg=3310)

    # Extract the buffers.
    boundary_buffers = perimeters_projected.boundary.buffer(distance)

    # Save.
    boundary_buffers_gdf = gpd.GeoDataFrame(
        perimeters_projected, geometry=boundary_buffers).to_crs(original_crs)

    if save:
        save_pickle(MTBS_BOUNDARY_BUFFER(distance), boundary_buffers_gdf)

    return boundary_buffers_gdf


# Trims the outside area around the fire, to exclude locations near the
# boundary to improve accuracy.
def trim_fire_area(
    distance: int,
    save: bool = True
) -> gpd.GeoDataFrame:
    sierra_perimeters = get_mtbs_perimeters_for_sierras()
    original_crs = sierra_perimeters.crs

    # Convert to a projected CRS.
    perimeters_projected = sierra_perimeters.to_crs(epsg=3310)

    # Extract the buffers around fire boundary.
    boundary_buffers = perimeters_projected.boundary.buffer(distance)

    # Trim area to exclude the area around fire perimeter.
    trimmed = perimeters_projected.difference(boundary_buffers)
    trimmed_gdf = gpd.GeoDataFrame(
        perimeters_projected, geometry=trimmed).to_crs(original_crs)

    if save:
        save_pickle(MTBS_PERIMETERS_TRIMMED(distance), trimmed_gdf)

    return trimmed_gdf
