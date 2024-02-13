import pandas as pd
from src.constants import INTERMEDIATE_RESULTS
from src.data.gedi import gedi_loader

INDEX = 'shot_number'
OVERLAYS_PATH = f"{INTERMEDIATE_RESULTS}/overlays"

DYNAMIC_WORLD = f"{OVERLAYS_PATH}/dynamic_world_overlay.pkl"
RECENT_LAND_COVER = f"{OVERLAYS_PATH}/recent_land_cover.pkl"
RECENT_LANDSAT = f"{OVERLAYS_PATH}/landsat_overlay.pkl"
DISTURBANCE_AGENTS = f"{OVERLAYS_PATH}/disturbances_overlay.pkl"
TCC = f"{OVERLAYS_PATH}/tree_canopy_cover_overlay.pkl"

# Overlays with fire datasets.
MTBS_BURN_BOUNDARIES = f"{OVERLAYS_PATH}/burn_boundary_overlay.pkl"
ALL_CALFIRE_FIRES = f"{OVERLAYS_PATH}/all_fires_overlay.pkl"
ALL_MTBS_FIRES = f"{OVERLAYS_PATH}/mtbs_fires_overlay_all.pkl"
# Includes only fires that occurred before GEDI shot was sampled, not after.
MTBS_FIRES = f"{OVERLAYS_PATH}/mtbs_fires_overlay.pkl"

ALL_MTBS_FIRES_SEVERITY = f"{OVERLAYS_PATH}/mtbs_severity_overlay_all.pkl"
MTBS_FIRES_SEVERITY = f"{OVERLAYS_PATH}/mtbs_severity_overlay.pkl"
MTBS_SEVERITY_CATEGORIES = f"{OVERLAYS_PATH}/mtbs_severity_categories_overlay.pkl"  # noqa: E501
MTBS_SEVERITY_WITH_PREFIRE_NDVI = f"{OVERLAYS_PATH}/mtbs_severity_overlay_with_ndvi.pkl"  # noqa: E501

TERRAIN = f"{OVERLAYS_PATH}/terrain_overlay.pkl"

NDVI_TIMESERIES = f"{OVERLAYS_PATH}/ndvi_timeseries_overlay.pkl"


def LANDCOVER(year):
    return f"{OVERLAYS_PATH}/land_cover_overlay_{year}.pkl"


def LANDSAT(year):
    return f"{OVERLAYS_PATH}/landsat_overlay_{year}.pkl"


def ADVANCED_LANDSAT(kind):
    return f"{OVERLAYS_PATH}/advanced_landsat_overlay_{kind}.pkl"


def MONTHLY_LANDSAT(year, month):
    return f"{OVERLAYS_PATH}/LANDSAT_{year}/monthly_landsat_overlay_{month}.pkl"  # noqa: E501


def MONTHLY_LANDSAT_FOLDER(year):
    return f"{OVERLAYS_PATH}/LANDSAT_{year}"


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
