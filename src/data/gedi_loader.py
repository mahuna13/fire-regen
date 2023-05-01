import geopandas as gpd

from src import constants
from src.data.gedi_database import GediDatabase
from src.utils.logging_util import get_logger

logger = get_logger(__file__)


def get_gedi_shots(
    geometry: gpd.GeoDataFrame,
    start_year: int = 2019,
    end_year: int = 2023,
    crs: str = constants.WGS84,
    save_file_path: str = None
):
    database = GediDatabase()

    # Load GEDI data within tile
    logger.info(
        f'Loading Level 4a GEDI shots for period {start_year}-{end_year} in this geometry')
    gedi_shots = database.query(
        table_name="level_4a",
        columns=[
            "shot_number",
            "absolute_time",
            "lon_lowestmode",
            "lat_lowestmode",
            "agbd",
            "agbd_pi_lower",
            "agbd_pi_upper",
            "agbd_se",
            "l2_quality_flag",
            "l4_quality_flag",
            "degrade_flag",
            "beam_type",
            "sensitivity",
            "pft_class"
        ],
        geometry=geometry,
        crs=crs,
        start_time=f"{start_year}-01-01",
        end_time=f"{end_year}-01-01",
    )
    logger.debug(
        f'Found {len(gedi_shots)} shots in {start_year}-{end_year} in the specified geometry')

    # Preliminary filtering to reduce computation size
    gedi_shots = gedi_shots[
        (gedi_shots.l2_quality_flag == 1)
        & (gedi_shots.l4_quality_flag == 1)
        & (gedi_shots.degrade_flag == 0)
    ]
    # Drop the columns we filtered on already.
    gedi_shots = gedi_shots.drop(
        columns=['l2_quality_flag', 'l4_quality_flag', 'degrade_flag'])

    logger.info(f'Number of GEDI shots found: {gedi_shots.shape[0]}')

    if save_file_path is not None:
        gedi_shots.to_csv(save_file_path)

    return gedi_shots


if __name__ == "__main__":
    # shape_gpd = gpd.read_file("data/shapefiles/seki_convex_hull.shp")
    shape_gpd = gpd.read_file("data/shapefiles/sierras_convex_hull.shp")
    get_gedi_shots(shape_gpd.geometry, 2019, 2023)
