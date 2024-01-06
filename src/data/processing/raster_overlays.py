import geopandas as gpd
import pandas as pd
from fastai.tabular.all import load_pickle, save_pickle
from src import constants
from src.data.processing import gedi_raster_matching
from src.utils.logging_util import get_logger

logger = get_logger(__file__)


OVERLAY_COLUMNS = ["shot_number", "longitude", "latitude"]


def get_gedi_shots(input_path: str) -> gpd.GeoDataFrame:
    gedi = load_pickle(input_path)

    return gpd.GeoDataFrame(gedi,
                            geometry=gpd.points_from_xy(gedi.longitude,
                                                        gedi.latitude),
                            crs=constants.WGS84)


# We will use LCMS dataset, because it's the easiest one to use through EE.
def overlay_land_cover(input_path: str, output_path: str, file_name: str):
    gedi_shots = load_pickle(input_path)
    START = 1985
    END = 2022

    for year in range(START, END):
        logger.info(f"Overlaying land cover for year {year}.")
        # LCSM is at 30m resolution, so we match raster on a 2x2 kernel.
        overlay = gedi_raster_matching.match_landcover(year, gedi_shots, 2)
        overlay["lc_year"] = year
        overlay = overlay.drop(columns=["longitude", "latitude",
                                        "absolute_time", "land_cover_mean"]
                               ).astype({"land_cover_median": "int"})

        # Save the intermediate result.
        save_pickle(f"{output_path}/{file_name}_{year}.pkl", overlay)


def overlay_terrain(input_path: str, output_path: str):
    gedi_shots = load_pickle(input_path)

    # Terrain dataset is 30m resolution, so we're matching each gedi shot with
    # a 2x2 terrain grid cells around the shot's coordinates.
    logger.info("Starting raster matching.")
    overlay = gedi_raster_matching.match_terrain(
        gedi_shots[OVERLAY_COLUMNS], kernel=2)

    return _save_overlay(overlay, output_path)


def overlay_landsat(input_path: str, output_path: str):
    gedi_shots = load_pickle(input_path)
    gedi_shots['gedi_year'] = gedi_shots.absolute_time.dt.year

    gedi_df_combined_years = []
    for year in range(2019, 2023):
        if year == 2022:
            gedi_for_year = gedi_shots[gedi_shots.gedi_year >=
                                       year][OVERLAY_COLUMNS]
        else:
            gedi_for_year = gedi_shots[gedi_shots.gedi_year ==
                                       year][OVERLAY_COLUMNS]

        logger.debug(f'Match Landsat for year {year}')
        raster = gedi_raster_matching.get_landsat_raster_sampler(year)
        gedi_for_year = gedi_raster_matching.sample_raster(raster,
                                                           gedi_for_year,
                                                           kernel=2)

        gedi_df_combined_years.append(gedi_for_year)

    overlay = pd.concat(gedi_df_combined_years)
    return _save_overlay(overlay, output_path)


def overlay_dynamic_world(input_path: str, output_path: str):
    gedi_shots = load_pickle(input_path)

    gedi_df_combined_years = []
    for year in range(2019, 2024):
        gedi_for_year = gedi_shots[gedi_shots.absolute_time.dt.year == year]

        logger.debug(f'Match with Dynamic World for year {year - 1}')
        raster = gedi_raster_matching.get_dw_raster_sampler(year - 1)

        # We match with a 3x3 kernel because DW resolution is 10m.
        gedi_for_year = gedi_raster_matching.sample_raster(raster,
                                                           gedi_for_year,
                                                           3)

        gedi_df_combined_years.append(gedi_for_year)

    overlay = pd.concat(gedi_df_combined_years)
    return _save_overlay(overlay, output_path)


def _save_overlay(df: pd.DataFrame, output_path: str):
    df = df.drop(columns=["longitude", "latitude"])

    logger.info("Saving the results.")
    save_pickle(output_path, df)
    return df
