import pandas as pd
from src.data.processing import gedi_raster_matching, overlay
from src.utils.logging_util import get_logger

logger = get_logger(__file__)


# We will use LCMS dataset, because it's the easiest one to use through EE.
def overlay_land_cover(df: pd.DataFrame, year: int):
    df = overlay.validate_input(df)
    logger.info(f"Overlaying land cover for year {year}.")
    # LCSM is at 30m resolution, so we match raster on a 2x2 kernel.
    result = gedi_raster_matching.match_landcover(year, df, 2)
    result["lc_year"] = year
    result = overlay.drop(columns=["longitude", "latitude",
                                   "absolute_time", "land_cover_mean"]
                          ).astype({"land_cover_median": "int"})

    return result


def overlay_terrain(df: pd.DataFrame):
    df = overlay.validate_input(df)

    # Terrain dataset is 30m resolution, so we're matching each gedi shot with
    # a 2x2 terrain grid cells around the shot's coordinates.
    logger.info("Starting raster matching.")
    result = gedi_raster_matching.match_terrain(df, kernel=2)

    # Keep median columns only and rename them.
    result.drop(columns=["aspect_mean", "aspect_std", "elevation_mean",
                         "elevation_std", "slope_mean", "slope_std",
                         "soil_mean", "soil_std"],
                inplace=True)

    result.rename(columns={
        "aspect_median": "aspect",
        "elevation_median": "elevation",
        "slope_median": "slope",
        "soil_median": "soil"
    }, inplace=True)

    return result


def overlay_landsat(df: pd.DataFrame):
    df = overlay.validate_input(df)
    gedi_df_combined_years = []
    for year in range(2019, 2023):
        if year == 2022:
            gedi_for_year = df[df.absolute_time.dt.year >= year]
        else:
            gedi_for_year = df[df.absolute_time.dt.year == year]

        logger.debug(f'Match Landsat for year {year}')
        raster = gedi_raster_matching.get_landsat_raster_sampler(year)
        gedi_for_year = gedi_raster_matching.sample_raster(raster,
                                                           gedi_for_year,
                                                           kernel=2)

        gedi_df_combined_years.append(gedi_for_year)

    result = pd.concat(gedi_df_combined_years)
    return result


def overlay_landsat_for_year(df: pd.DataFrame, year: int):
    df = overlay.validate_input(df)
    LANDSAT_COLUMNS = gedi_raster_matching.get_landsat_bands(year)

    logger.debug(f'Match Landsat for year {year}')
    raster = gedi_raster_matching.get_landsat_raster_sampler(year)
    matched = gedi_raster_matching.sample_raster(raster,
                                                 df,
                                                 kernel=2)
    for column in LANDSAT_COLUMNS:
        df[f"{column}_{year}"] = matched[f"{column}_mean"]

    return df


def overlay_ndvi(df: pd.DataFrame):
    df = overlay.validate_input(df)
    for year in range(1984, 2023):
        logger.debug(f'Match NDVI for year {year}')

        COL_NAME = f"ndvi_{year}"
        raster = gedi_raster_matching.get_ndvi_raster_sampler(year)
        matched = gedi_raster_matching.sample_raster(raster, df, 2) \
            .rename(columns={"ndvi_mean": COL_NAME})

        df[COL_NAME] = matched[COL_NAME]

    return df


def overlay_dynamic_world(df: pd.DataFrame):
    df = overlay.validate_input(df)

    gedi_df_combined_years = []
    for year in range(2019, 2024):
        gedi_for_year = df[df.absolute_time.dt.year == year]

        logger.debug(f'Match with Dynamic World for year {year - 1}')
        raster = gedi_raster_matching.get_dw_raster_sampler(year - 1)

        # We match with a 3x3 kernel because DW resolution is 10m.
        gedi_for_year = gedi_raster_matching.sample_raster(raster,
                                                           gedi_for_year,
                                                           3)

        gedi_df_combined_years.append(gedi_for_year)

    result = pd.concat(gedi_df_combined_years)
    return result


def overlay_tree_cover(df: pd.DataFrame):
    df = overlay.validate_input(df)

    for year in [2000, 2005, 2010, 2015]:
        logger.debug(f'Match with Global Tree Canopy Cover for year {year}')
        raster = gedi_raster_matching.get_gfcc_raster_sampler(year)

        COL_NAME = f"tcc_{year}"
        # We match with a 2x2 kernel because GFCC resolution is 30m.
        matched = gedi_raster_matching.sample_raster(raster, df, 2) \
            .rename(columns={"tree_canopy_cover_mean": COL_NAME})

        df[COL_NAME] = matched[COL_NAME]

    return df
