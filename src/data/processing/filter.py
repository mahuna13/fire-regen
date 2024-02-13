import geopandas as gpd
import pandas as pd
from fastai.tabular.all import load_pickle
from src.constants import SIERRAS
from src.data.processing import overlay
from src.utils.logging_util import get_logger

logger = get_logger(__file__)


def exclude_shots_outside_sierra_conservancy(
    df: pd.DataFrame
):
    SIERRAS_ROI = gpd.read_file(SIERRAS)[["geometry"]]
    # Keep only shots that fall within the Sierra conservancy teritory.
    return df.sjoin(SIERRAS_ROI, how="inner", predicate="within").drop(
        columns="index_right")


def exclude_steep_slopes(
    df: pd.DataFrame
):
    return df[(df.slope < 30) & (df.elevation_difference_tdx.abs() < 50)]


def divide_based_on_burn_occurrence(df: pd.DataFrame):
    # burned, burned once, and unburned - report length.
    burned_once = df[df.fire_count == 1]
    burned_multiple = df[df.fire_count > 1]
    unburned = df[df.fire_count == 0]
    return burned_once, burned_multiple, unburned


def filter_other_disturbances(
        burned: pd.DataFrame,
        unburned: pd.DataFrame):
    DA_CLASSES = [2, 3, 5]

    # Join with disturbances.
    da = load_pickle(overlay.DISTURBANCE_AGENTS)
    unburned_da = unburned.join(da, how="left")
    burned_da = burned.join(da, how="left")

    match_unburned = unburned_da[
        (unburned_da.da_year < unburned_da.absolute_time.dt.year) &
        (unburned_da.da_min.isin(DA_CLASSES))]
    match_burned = burned_da[
        (burned_da.da_year - 1 > burned_da.fire_ig_date.dt.year) &
        (burned_da.da_year < burned_da.absolute_time.dt.year) &
        (burned_da.da_min.isin(DA_CLASSES))]

    match_unburned_index = match_unburned.index.drop_duplicates()
    match_burned_index = match_burned.index.drop_duplicates()

    unburned_filtered = unburned[~unburned.index.isin(match_unburned_index)]
    burned_filtered = burned[~burned.index.isin(match_burned_index)]

    return burned_filtered, unburned_filtered


def filter_burned_based_on_land_cover(df: pd.DataFrame):
    start = 1985
    end = 2022

    all_years = []
    for year in range(start, end):
        lc_year = max(1985, year - 1)
        fire_year_df = df[df.fire_ig_date.dt.year == year]
        lc_df = load_pickle(overlay.LANDCOVER(lc_year))

        all_years.append(filter_for_land_cover(fire_year_df, lc_df))

    return pd.concat(all_years)


def filter_based_on_recent_land_cover(
        df: pd.DataFrame):
    recent_lc = load_pickle(overlay.RECENT_LAND_COVER).drop(
        columns=["absolute_time"])
    return filter_for_land_cover(df, recent_lc)


def filter_for_land_cover(
        df_input: pd.DataFrame,
        land_cover_df: pd.DataFrame):
    filtered = df_input.join(
        land_cover_df[["land_cover_std", "land_cover_median"]], how="left")
    # Filter for all pixels being "trees"
    filtered = filtered[(filtered.land_cover_std == 0)
                        & (filtered.land_cover_median == 1)]

    return df_input.loc[filtered.index]


def filter_for_land_cover_in_year(
        year: int,
        df_input: pd.DataFrame):
    lc_df = load_pickle(overlay.LANDCOVER(year))
    return filter_for_land_cover(df_input, lc_df)


def filter_burn_severity(df: pd.DataFrame):
    INVALID = -5000

    # Filter invalid dnbr values.
    df = df[df.dnbr_min > INVALID]
    return df


def exclude_fire_boundaries(df: pd.DataFrame):
    # Get rid of all the shots that fell around the boundary of any fire.
    boundary_shots = load_pickle(overlay.MTBS_BURN_BOUNDARIES)
    return df[~df.index.isin(boundary_shots.index)]


def filter_burn_areas(df: pd.DataFrame):
    logger.info("Number of rows before filtering.")

    # Get fires from call fires.
    all_fires = load_pickle(overlay.ALL_CALFIRE_FIRES)

    # Eliminate all shots that fall within small and recent fires.
    small_recent = all_fires[(all_fires.Shape_Area_Acres < 1000)
                             & ((all_fires.YEAR_ >= 1984))]
    df = df[~df.index.isin(small_recent.index)]
    logger.info(f"Number of rows after filtering small fires: {len(df)}")

    # Eliminate all shots that burned 10 years before 1984.
    old_fires = all_fires[(all_fires.YEAR_ < 1984) & (all_fires.YEAR_ >= 1974)]
    df = df[~df.index.isin(old_fires.index)]
    logger.info(f"Number of rows after filtering old fires: {len(df)}")
    return df
