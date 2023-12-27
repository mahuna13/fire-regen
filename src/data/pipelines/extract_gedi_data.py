import geopandas as gpd
from src.constants import GEDI_INTERMEDIATE_PATH, USER_PATH
from src.data.gedi import gedi_loader

SIERRAS = gpd.read_file(f"{USER_PATH}/data/shapefiles/sierras_convex_hull.shp")
SEKI = gpd.read_file(f"{USER_PATH}/data/shapefiles/seki_convex_hull.shp")

SIERRAS_GEDI_ALL_COLUMNS = f"{GEDI_INTERMEDIATE_PATH}/sierras_gedi_shots.pkl"
SEKI_GEDI_ALL_COLUMNS = f"{GEDI_INTERMEDIATE_PATH}/seki_gedi_shots.pkl"

SIERRAS_GEDI_ID_COLUMNS = f"{GEDI_INTERMEDIATE_PATH}/gedi_info_columns.pkl"
SEKI_GEDI_ID_COLUMNS = f"{GEDI_INTERMEDIATE_PATH}/seki_gedi_info_columns.pkl"


if __name__ == '__main__':
    # Fetch GEDI shots from postgress for the regions of interest, and save
    # them in pkl files.
    gedi_loader.fetch_gedi_from_postgres(
        SIERRAS.geometry, SIERRAS_GEDI_ALL_COLUMNS)

    gedi_loader.fetch_gedi_from_postgres(
        SEKI.geometry, SEKI_GEDI_ALL_COLUMNS)

    # Extract only a subset of ID GEDI columns for the dataset in separate pkl
    # files. We do this to do subsequent data processing on data frames that
    # are as small as possible, to minimize processing time.
    #
    # These dataframes can always be joined with complete GEDI data above by
    # joining on 'shot_number' column.
    gedi_loader.extract_and_save_id_columns(
        SIERRAS_GEDI_ALL_COLUMNS, SIERRAS_GEDI_ID_COLUMNS)

    gedi_loader.extract_and_save_id_columns(
        SEKI_GEDI_ALL_COLUMNS, SEKI_GEDI_ID_COLUMNS)
