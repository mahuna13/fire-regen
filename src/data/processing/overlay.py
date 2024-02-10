import pandas as pd
from src.constants import INTERMEDIATE_RESULTS
from src.data.gedi import gedi_loader

INDEX = 'shot_number'
OVERLAYS_PATH = f"{INTERMEDIATE_RESULTS}/overlays"

DYNAMIC_WORLD = f"{OVERLAYS_PATH}/dynamic_world_overlay.pkl"
RECENT_LAND_COVER = f"{OVERLAYS_PATH}/recent_land_cover.pkl"
DISTURBANCE_AGENTS = f"{OVERLAYS_PATH}/disturbances_overlay.pkl"
TCC = f"{OVERLAYS_PATH}/tree_canopy_cover_overlay.pkl"


def get_overlays_path(file_name: str):
    return f"{OVERLAYS_PATH}/{file_name}"


def validate_input(df: pd.DataFrame):
    if df.index.name != INDEX:
        # Attempt to set the index.
        df.set_index(INDEX, inplace=True)

    id_columns = gedi_loader.ID_COLUMNS[:]
    id_columns.remove(INDEX)

    if not (all(column in df.columns for column in id_columns)):
        raise Exception(
            f"Some columns from {id_columns} are \
            missing from the dataframe {df.columns}")

    return df
