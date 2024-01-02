import geopandas as gpd
import pandas as pd
from fastai.tabular.all import load_pickle, save_pickle
from src import constants
from src.data.gedi.gedi_database import GediDatabase
from src.utils.logging_util import get_logger

logger = get_logger(__file__)


# All columns relevant for combined analysis.
ALL_POSTGRES_COLUMNS = [
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
]


# Minimal columns necessary for intersecting GEDI data with other raster-based
# datasets.
ID_COLUMNS = [
    "shot_number",
    "longitude",
    "latitude",
    "absolute_time"
]


def get_combined_l2ab_l4a_shots(
    geometry: gpd.GeoDataFrame,
    start_year: int = 2019,
    end_year: int = 2024,
    crs: str = constants.WGS84,
    save_file_path: str = None
):
    database = GediDatabase()

    # Load GEDI data within tile
    logger.info(f'Loading combined GEDI shots for period \
        {start_year}-{end_year} in this geometry')
    gedi_shots = database.query(
        table_name="filtered_l2ab_l4a_shots",
        columns=ALL_POSTGRES_COLUMNS,
        geometry=geometry,
        crs=crs,
        start_time=f"{start_year}-01-01",
        end_time=f"{end_year}-01-01",
    )
    logger.debug(f'Found {len(gedi_shots)} shots in \
        {start_year}-{end_year} in the specified geometry')

    # Preliminary filtering to reduce computation size
    gedi_shots = gedi_shots[
        (gedi_shots.l4_quality_flag == 1)
        & (gedi_shots.l4_algorithm_run_flag == 1)
    ]
    # Drop the columns we filtered on already.
    gedi_shots = gedi_shots.drop(
        columns=['l4_algorithm_run_flag', 'l4_quality_flag'])

    gedi_shots = initial_shot_processing(gedi_shots)

    logger.info(f'Number of GEDI shots found: {gedi_shots.shape[0]}')

    logger.info(save_file_path)
    if save_file_path is not None:
        logger.info(f"Saving data into a pickle file: {save_file_path}")
        save_pickle(save_file_path, gedi_shots)
        logger.info("Data successfully saved.")

    return gedi_shots


def initial_shot_processing(gedi_gdf: pd.DataFrame):
    gedi_gdf.absolute_time = pd.to_datetime(
        gedi_gdf.absolute_time, utc=True, format='mixed')

    # Extract month and year of when GEDI shot was taken.
    gedi_gdf['gedi_year'] = gedi_gdf.absolute_time.dt.year
    gedi_gdf['gedi_month'] = gedi_gdf.absolute_time.dt.month

    gedi_gdf.rename(columns={"lon_lowestmode": "longitude",
                             "lat_lowestmode": "latitude"}, inplace=True)
    return gedi_gdf


def fetch_gedi_from_postgres(
        geometry: gpd.GeoDataFrame,
        output_path: str):

    return get_combined_l2ab_l4a_shots(
        geometry=geometry,
        save_file_path=output_path
    )


# Load a dataframe with all GEDI columns, and extract only the ID and
# geolocation columns, to minimize processing time for further processing.
def extract_and_save_id_columns(input_path: str, output_path: str):
    gedi_shots = load_pickle(input_path)
    save_pickle(output_path, gedi_shots[ID_COLUMNS])
