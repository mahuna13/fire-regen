from src.data import gedi_loader, gedi_raster_matching
from src.constants import DATA_PATH, USER_PATH
from src.utils.logging_util import get_logger
import geopandas as gpd
import pandas as pd
import numpy as np
from fastai.tabular.all import save_pickle, load_pickle
pd.options.mode.chained_assignment = None  # default='warn'


logger = get_logger(__file__)

GEDI_PATH = f"{DATA_PATH}/gedi_intermediate"


def add_time_since_burn_categories(df: gpd.GeoDataFrame):
    c = pd.cut(
        df[['time_since_burn']].stack(),
        [-np.inf, 0, 10, 20, 30, np.inf],
        labels=['unburned', 'burn_10', 'burn_20', "burn_30", "burn_40"]
    )
    return df.join(c.unstack().add_suffix('_cat'))


def add_time_since_burn_categories_3(df: gpd.GeoDataFrame):
    c = pd.cut(
        df[['time_since_burn']].stack(),
        range(0, 37, 3),
        labels=['burn_3', 'burn_6', 'burn_9',
                "burn_12", "burn_15", "burn_18", "burn_21",
                "burn_24", "burn_27", "burn_30", "burn_33", "burn_36"]
    )
    return df.join(c.unstack().add_suffix('_cat_3'))


def add_time_since_burn_categories_5(df: gpd.GeoDataFrame):
    c = pd.cut(
        df[['time_since_burn']].stack(),
        range(0, 37, 5),
        labels=['burn_5', 'burn_10', 'burn_15',
                "burn_20", "burn_25", "burn_30", "burn_35"]
    )
    return df.join(c.unstack().add_suffix('_cat_5'))


def add_time_since_burn_categories_7(df: gpd.GeoDataFrame):
    c = pd.cut(
        df[['time_since_burn']].stack(),
        range(0, 37, 7),
        labels=['burn_7', 'burn_14', 'burn_21',
                "burn_28", "burn_35"]
    )
    return df.join(c.unstack().add_suffix('_cat_7'))


def add_time_since_burn_categories_10(df: gpd.GeoDataFrame):
    c = pd.cut(
        df[['time_since_burn']].stack(),
        range(0, 41, 10),
        labels=['burn_10', 'burn_20', "burn_30", "burn_40"]
    )
    return df.join(c.unstack().add_suffix('_cat_10'))


def print_burn_stats(df):
    # Gedi statistics
    unburned_ratio = (df[df.burn_counts_median == 0].shape[0])/df.shape[0]
    high_burn_ratio = (df[(df.burn_severity_median == 4)].shape[0])/df.shape[0]
    medium_burn_ratio = (
        df[(df.burn_severity_median == 3)].shape[0])/df.shape[0]
    low_burn_ratio = (df[(df.burn_severity_median == 2)].shape[0])/df.shape[0]

    print(f'Unburned ratio: {unburned_ratio*100}%')
    print(f'High-burned ratio: {high_burn_ratio*100}%')
    print(f'Medium-burned ratio: {medium_burn_ratio*100}%')
    print(f'Low-burned ratio: {low_burn_ratio*100}%')


def get_severity(df, severity):
    return df[df.severity == severity]


'''
Stage 7 of the GEDI pipeline - Filter Land Cover.
'''


def load_stage_7(kernel: int):
    gedi_burned = get_gedi_as_gdp(
        f"sierras_combined_shots_stage_7_{kernel}x{kernel}_burned.pkl")

    _, gedi_unburned = load_stage_5(kernel)

    return gedi_burned, gedi_unburned


def stage_7_filter_land_cover_l4a_sierras(kernel: int, save: bool = True):
    gedi_burned, gedi_unburned = load_stage_6(kernel)

    # Keep only trees.
    gedi_burned = gedi_burned[
        (gedi_burned.land_cover_std == 0) &
        (gedi_burned.land_cover_median == 1)]
    logger.debug(f'Number of shots that were trees before they burned: \
                {gedi_burned.shape[0]}')

    # Keep only unburned trees.
    # gedi_unburned = gedi_unburned[
    #    (gedi_unburned.land_cover_std == 0) &
    #    (gedi_unburned.land_cover_median == 1)]
    # logger.debug(f'Number of unburned shots that were trees in 2021: \
    #        {gedi_unburned.shape[0]}')

    # Drop land cover columns, since we've filtered them.
    columns_to_drop = [f"land_cover_{kernel}x{kernel}",
                       "land_cover_mean",
                       "land_cover_std",
                       "land_cover_median"]

    gedi_burned.drop(columns=columns_to_drop, inplace=True)
    # gedi_unburned.drop(columns=columns_to_drop, inplace=True)

    if save:
        _save_df_as_pickle("sierras_combined_shots_stage_7_{kernel}x{kernel}_burned.pkl",
                           gedi_burned)

        # _save_df_as_pickle("sierras_combined_shots_stage_7_{kernel}x{kernel}_unburned.pkl",
        #    gedi_unburned)

    return gedi_burned, gedi_unburned


'''
Stage 6 of the GEDI pipeline - Match Land Cover.
'''


def load_stage_6(kernel: int):
    gedi_burned = get_gedi_as_gdp(
        f"sierras_combined_shots_stage_6_{kernel}x{kernel}_burned.pkl")

    gedi_unburned = get_gedi_as_gdp(
        f"sierras_combined_shots_stage_5_{kernel}x{kernel}_unburned.pkl")

    return gedi_burned, gedi_unburned


def stage_6_match_land_cover_l4a_sierras(kernel: int, save: bool = True):
    gedi_burned, gedi_unburned = load_stage_5(kernel)

    # For each burn year, we want to match it to the land cover of the previous
    # year.
    gedi_burned = gedi_raster_matching.match_burn_landcover(
        gedi_burned, kernel)

    # TODO: Decide whether to match unburned to anything here, or just leave
    # it as is. We will later match it to Dynamic World cover, so it's likely
    # we can just use that for filtering.
    # Match unburned pixels with the latest land cover.
    # gedi_unburned = gedi_raster_matching.match_landcover_for_year(
    #    2022, gedi_unburned, 3)

    if save:
        _save_df_as_pickle("sierras_combined_shots_stage_6_{kernel}x{kernel}_burned.pkl",
                           gedi_burned)

        # _save_df_as_pickle("sierras_combined_shots_stage_6_{kernel}x{kernel}_unburned.pkl",
        #    gedi_unburned)

    return gedi_burned, gedi_unburned


'''
Stage 5 of the GEDI pipeline - Filter for regrowth analysis.
'''


def load_stage_5(kernel: int):
    gedi_burned = get_gedi_as_gdp(
        f"sierras_combined_shots_stage_5_{kernel}x{kernel}_burned.pkl")

    gedi_unburned = get_gedi_as_gdp(
        f"sierras_combined_shots_stage_5_{kernel}x{kernel}_unburned.pkl")

    return gedi_burned, gedi_unburned


def stage_5_filter_for_regrowth_l4a_sierras(kernel: int, save: bool = True):
    gedi_burned, gedi_unburned = load_stage_4(kernel)

    gedi_burned = filter_shots_for_regrowth_analysis(gedi_burned)

    if save:
        _save_df_as_pickle("sierras_combined_shots_stage_5_{kernel}x{kernel}_burned.pkl",
                           gedi_burned)
        _save_df_as_pickle("sierras_combined_shots_stage_5_{kernel}x{kernel}_unburned.pkl",
                           gedi_unburned)

    return gedi_burned, gedi_unburned


def filter_shots_for_regrowth_analysis(gedi_gdf: gpd.GeoDataFrame):
    # Get rid of shots where the GEDI shot happened before fire, since we're
    # only looking at recovery.
    gedi_gdf = gedi_gdf[gedi_gdf.time_since_burn > 0]
    logger.debug(f'Number of shots that happened after fires: \
                   {gedi_gdf.shape[0]}')

    # Only look at 2-4 burn severity categories.
    gedi_gdf = gedi_gdf[gedi_gdf.severity.isin([2, 3, 4])]
    logger.debug(f'Number of shots that burned in 2-4 categories: \
                   {gedi_gdf.shape[0]}')

    return gedi_gdf


'''
Stage 4 of the GEDI pipeline - Filter for burned areas.
'''


def load_stage_4(kernel: int):
    gedi_burned = get_gedi_as_gdp(
        f"sierras_combined_shots_stage_4_{kernel}x{kernel}_burned.pkl")

    gedi_unburned = get_gedi_as_gdp(
        f"sierras_combined_shots_stage_4_{kernel}x{kernel}_unburned.pkl")

    return gedi_burned, gedi_unburned


def stage_4_filter_burns_l4a_sierras(kernel: int, save: bool = True):
    logger.debug("Read in intermediate data from stage 3.")
    gedi = load_stage_3(kernel)

    logger.debug("Filter burn areas.")
    gedi_burned, gedi_unburned = filter_burn_areas(gedi, kernel)

    if save:
        _save_df_as_pickle("sierras_combined_shots_stage_4_{kernel}x{kernel}_burned.pkl",
                           gedi_burned)
        _save_df_as_pickle("sierras_combined_shots_stage_4_{kernel}x{kernel}_unburned.pkl",
                           gedi_unburned)


def filter_burn_areas(gedi: gpd.GeoDataFrame, kernel: int):
    gedi_burned, gedi_unburned = divide_shots_into_burned_and_unburned(gedi)
    gedi_burned = exclude_shots_on_burn_boundaries(gedi_burned)

    # Remove (kernel x kernel) columns, as they are not relevant any more
    # since all shots at the boundary have been excluded.
    columns_to_drop = [f"burn_severity_{kernel}x{kernel}",
                       f"burn_counts_{kernel}x{kernel}",
                       f"burn_year_{kernel}x{kernel}",
                       "burn_severity_mean",
                       "burn_counts_mean",
                       "burn_year_mean",
                       "burn_severity_std",
                       "burn_counts_std",
                       "burn_year_std"]

    columns_to_rename = {"burn_severity_median": "severity",
                         "burn_counts_median": "burn_count",
                         "burn_year_median": "burn_year"}

    gedi_burned.drop(columns=columns_to_drop, inplace=True)
    gedi_unburned.drop(columns=columns_to_drop, inplace=True)
    gedi_burned.rename(columns=columns_to_rename, inplace=True)
    gedi_unburned.rename(columns=columns_to_rename, inplace=True)
    return gedi_burned, gedi_unburned


def divide_shots_into_burned_and_unburned(gedi_gdf: gpd.GeoDataFrame):
    # Get rid of all the shots that are on the burn boundaries between burned
    # and unburned, or burned once and burned multiple times.
    gedi_gdf = gedi_gdf[
        (gedi_gdf.burn_counts_std == 0) &
        (gedi_gdf.burn_year_std == 0)]
    logger.debug(f'Excluded shots on the burn boundaries, shots remaining: \
        {gedi_gdf.shape[0]}')

    # Divide GEDI shots into burned and unburned.
    gedi_burned = gedi_gdf[gedi_gdf.burn_counts_median > 0]
    logger.debug(f'Number of GEDI shots that burned at least once: \
                 {gedi_burned.shape[0]}')

    gedi_unburned = gedi_gdf[
        (gedi_gdf.burn_counts_median == 0) &
        (gedi_gdf.burn_severity_median == 0) &
        (gedi_gdf.burn_severity_std == 0)]
    logger.debug(f'Number of GEDI shots that never burned since 1984: \
        {gedi_unburned.shape[0]}')

    # For the burned shots, calculate time in years since they burned.
    gedi_burned['time_since_burn'] = gedi_burned.gedi_year - \
        gedi_burned.burn_year_median

    # For unburned, just set the time to -1.
    gedi_unburned['time_since_burn'] = -1

    return gedi_burned, gedi_unburned


def exclude_shots_on_burn_boundaries(df: gpd.GeoDataFrame):
    # Only look at pixels that burned exactly once.
    # TODO: consider if we want to filter for exactly the same burn at this
    # point, or later. It would be interesting just to take a look at the shots
    # that burned multiple times.

    # For now, we leave this commented out.

    # df = df[df.burn_counts_median == 1]
    # logger.debug(f'Number of shots that burned exactly once: \
    #               {df.shape[0]}')

    df = df[df.burn_severity_std == 0]
    logger.debug(f'Number of GEDI shots that have a perfect match with burn \
                   raster (all surrounding pixels have the same severity): \
                   {df.shape[0]}')

    return df


def _save_df_as_pickle(file_name: str, df):
    logger.debug(f"Saving {file_name}")
    save_pickle(f"{GEDI_PATH}/{file_name}", df)
