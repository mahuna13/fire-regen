import geopandas as gpd
from src.data import fire_perimeters
from src.constants import DATA_PATH, USER_PATH
from fastai.tabular.all import load_pickle, save_pickle


# Fetch simplified regions of interest.
SEKI = gpd.read_file(f"{USER_PATH}/data/shapefiles/seki_convex_hull.shp")
SIERRAS = gpd.read_file(f"{USER_PATH}/data/shapefiles/sierras_convex_hull.shp")


def MTBS_PERIMETERS_TRIMMED(distance):
    return f"{DATA_PATH}/mtbs/perimeters_{distance}m_trimmed.pkl"


MTBS_FIRES_TRIMMED_10m = MTBS_PERIMETERS_TRIMMED(10)
MTBS_INDIVIDUAL_FIRES = f"{DATA_PATH}/mtbs/all_fires/mtbs"


def get_mtbs_perimeters_for_sierras():
    mtbs_db = fire_perimeters.MTBSFirePerimetersDB(SIERRAS)

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
