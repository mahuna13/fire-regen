import geopandas as gpd

from src import constants
from src.data.gedi_database import GediDatabase
from src.utils.logging_util import get_logger
from fastai.tabular.all import save_pickle

logger = get_logger(__file__)


def get_combined_l2ab_l4a_shots(
    geometry: gpd.GeoDataFrame,
    start_year: int = 2019,
    end_year: int = 2024,
    crs: str = constants.WGS84,
    save_file_path: str = None
):
    database = GediDatabase()

    # Load GEDI data within tile
    logger.info(
        f'Loading combined GEDI shots for period {start_year}-{end_year} in this geometry')
    gedi_shots = database.query(
        table_name="filtered_l2ab_l4a_shots",
        columns=[
            # shot information
            "shot_number",
            "beam_type",
            # Temporal
            "absolute_time",

            # Geolocation
            "lon_lowestmode",
            "lat_lowestmode",
            "elevation_difference_tdx",

            # measurements
            "agbd",
            "agbd_se",
            "fhd_normal",
            "pai",
            "pai_z",
            "pavd_z",
            "rh_98",
            "rh_70",
            "rh_50",
            "rh_25",
            "cover",
            "cover_z",

            # Quality Data
            "sensitivity_a0",
            "l4_algorithm_run_flag",
            "l4_quality_flag",
            # "predictor_limit_flag",
            # "response_limit_flag",
            "solar_elevation",

            # Processing data
            # "selected_algorithm",
            # "selected_mode"

            # Land cover
            "gridded_pft_class",
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
        (gedi_shots.l4_quality_flag == 1)
        & (gedi_shots.l4_algorithm_run_flag == 1)
    ]
    # Drop the columns we filtered on already.
    gedi_shots = gedi_shots.drop(
        columns=['l4_algorithm_run_flag', 'l4_quality_flag'])

    logger.info(f'Number of GEDI shots found: {gedi_shots.shape[0]}')

    if save_file_path is not None:
        save_pickle(save_file_path, gedi_shots)

    return gedi_shots


def get_l2b_gedi_shots(
    geometry: gpd.GeoDataFrame,
    start_year: int = 2019,
    end_year: int = 2024,
    crs: str = constants.WGS84,
    save_file_path: str = None
):
    database = GediDatabase()

    # Load GEDI data within tile
    logger.info(
        f'Loading Level 2B GEDI shots for period {start_year}-{end_year} in this geometry')
    gedi_shots = database.query(
        table_name="level_2b",
        columns=[
            "shot_number",
            "absolute_time",
            "lon_lowestmode",
            "lat_lowestmode",
            "fhd_normal",
            "pai",
            "pai_z",
            "pavd_z",
            "rh100",
            "cover",
            "cover_z",
            "l2a_quality_flag",
            "l2b_quality_flag",
            "degrade_flag",
            "beam_type",
            "sensitivity",
            "gridded_pft_class",
            "surface_flag",
            "elev_lowestmode",
            "geolocation/digital_elevation_model",
            "land_cover_data/pft_class"
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
        (gedi_shots.l2b_quality_flag == 1)
        & (gedi_shots.degrade_flag == 0)
    ]
    # Drop the columns we filtered on already.
    gedi_shots = gedi_shots.drop(
        columns=['l2b_quality_flag', 'degrade_flag'])

    logger.info(f'Number of GEDI shots found: {gedi_shots.shape[0]}')

    if save_file_path is not None:
        save_pickle(save_file_path, gedi_shots)

    return gedi_shots


def get_l4a_gedi_shots(
    geometry: gpd.GeoDataFrame,
    start_year: int = 2019,
    end_year: int = 2024,
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
        save_pickle(save_file_path, gedi_shots)

    return gedi_shots
