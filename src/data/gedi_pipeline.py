from src.utils.logging_util import get_logger
import seaborn as sns
import geopandas as gpd
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'


logger = get_logger(__file__)


def get_gedi_as_gdp(csv_file_path: str) -> gpd.GeoDataFrame:
    gedi = pd.read_csv(csv_file_path, index_col=0)
    return gpd.GeoDataFrame(gedi,
                            geometry=gpd.points_from_xy(gedi.lon_lowestmode,
                                                        gedi.lat_lowestmode),
                            crs=4326)


def process_shots(gedi_gdf: gpd.GeoDataFrame):
    gedi_gdf.absolute_time = pd.to_datetime(gedi_gdf.absolute_time, utc=True)

    # Extract month and year of when GEDI shot was taken.
    gedi_gdf['gedi_year'] = gedi_gdf.absolute_time.dt.year
    gedi_gdf['gedi_month'] = gedi_gdf.absolute_time.dt.month

    return gedi_gdf


def filter_shots(gedi_gdf: gpd.GeoDataFrame):
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
        (gedi_gdf.burn_severity_median == 0)]
    logger.debug(f'Number of GEDI shots that never burned since 1984: \
        {gedi_unburned.shape[0]}')

    # For the burned shots, calculate time in years since they burned.
    gedi_burned['time_since_burn'] = gedi_burned.gedi_year - \
        gedi_burned.burn_year_median

    # For unburned, just set the time to -1.
    gedi_unburned['time_since_burn'] = -1

    return gedi_burned, gedi_unburned


def filter_shots_for_regrowth_analysis(gedi_gdf: gpd.GeoDataFrame):
    # Get rid of shots where the GEDI shot happened before fire, since we're
    # only looking at recovery.
    gedi_gdf = gedi_gdf[gedi_gdf.time_since_burn > 0]
    logger.debug(f'Number of shots that happened after fires: \
                   {gedi_gdf.shape[0]}')

    # Only look at pixels that burned exactly once.
    gedi_gdf = gedi_gdf[gedi_gdf.burn_counts_median == 1]
    logger.debug(f'Number of shots that burned exactly once: \
                   {gedi_gdf.shape[0]}')

    # Only look at 2-4 burn severity categories.
    gedi_gdf = gedi_gdf[gedi_gdf.burn_severity_median.isin([2, 3, 4])]
    logger.debug(f'Number of shots that burned in 2-4 categories: \
                   {gedi_gdf.shape[0]}')

    gedi_gdf_perfect = gedi_gdf[gedi_gdf.burn_severity_std == 0]
    logger.debug(f'Number of GEDI shots that have a perfect match with burn \
                   raster (all 3x3 pixels have the same severity): \
                   {gedi_gdf_perfect.shape[0]}')

    return gedi_gdf_perfect


def filter_for_trees(gedi_gdf: gpd.GeoDataFrame):
    return gedi_gdf[(gedi_gdf.land_cover_std == 0) &
                    (gedi_gdf.land_cover_median == 1)]


def add_time_since_burn_categories(df: gpd.GeoDataFrame):
    c = pd.cut(
        df[['time_since_burn']].stack(),
        [-np.inf, 0, 10, 20, 30, np.inf],
        labels=['unburned', 'burn_10', 'burn_20', "burn_30", "burn_40"]
    )
    return df.join(c.unstack().add_suffix('_cat'))