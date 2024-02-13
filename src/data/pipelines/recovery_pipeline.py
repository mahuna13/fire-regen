import pandas as pd
from fastai.tabular.all import load_pickle, save_pickle
from src.constants import INTERMEDIATE_RESULTS
from src.data.adapters import mtbs
from src.data.pipelines.extract_gedi_data import SIERRAS_GEDI_ALL_COLUMNS
from src.data.processing import filter, overlay
from src.data.utils import gedi_utils
from src.utils.logging_util import get_logger

pd.options.mode.chained_assignment = None

logger = get_logger(__file__)

OUTPUT_PATH = f"{INTERMEDIATE_RESULTS}/pipelines/recovery"

COLUMNS = ['longitude', 'latitude', 'absolute_time', 'fire_size_acres',
           'fire_name', 'fire_ig_date', 'days_since_fire', 'pre_fire_ndvi',
           'severity', 'YSF', 'YSF_cat_5', 'agbd', 'cover', 'fhd_normal',
           'pai', 'pai_z', 'rh_25', 'rh_50', 'rh_70', 'rh_98', 'aspect',
           'elevation', 'slope', 'soil', 'geometry']


def filter_unburned_based_on_pre_fire_landcover():
    unburned = load_pickle(f"{OUTPUT_PATH}/unburned.pkl")

    for year in range(1985, 2022):
        logger.info(f"Filter land cover for year: {year}.")
        unburned_lc = filter.filter_for_land_cover_in_year(year, unburned)
        logger.info(f"Number of shots remaining: {len(unburned_lc)}.")
        save_pickle(f"{OUTPUT_PATH}/unburned_lc_{year}.pkl", unburned_lc)


def run():
    shots = load_pickle(overlay.MTBS_SEVERITY_WITH_PREFIRE_NDVI)
    logger.info(f"Number of input shots: {len(shots)}.")

    # Include only shots that fall within Sierra Conservancy
    shots = filter.exclude_shots_outside_sierra_conservancy(shots)
    logger.info(f"Number of shots within Sierras Conservancy: {len(shots)}.")

    # Get rid of all the shots that fell around the boundary of any fire.
    shots = filter.exclude_fire_boundaries(shots)
    logger.info(f"Number of shots after filtering for areas around \
        fire boundaries: {len(shots)}")

    shots = filter.filter_burn_areas(shots)
    burned, _, unburned = filter.divide_based_on_burn_occurrence(shots)
    logger.info(f"Number of shots that burned once: {len(burned)}")
    logger.info(f"Number of shots that never burned: {len(unburned)}")

    # Only keep shots that were labeled as "trees" before they burned.
    burned = filter.filter_burned_based_on_land_cover(burned)
    logger.info(
        f"Number of shots that were forest before they burned: {len(burned)}")

    # Filter other disturbances
    burned, unburned = filter.filter_other_disturbances(burned, unburned)
    logger.info(
        f"Burned shots after filtering for disturbances: {len(burned)}")
    logger.info(
        f"Unburned shots after filtering for disturbances: {len(unburned)}")

    # Determine severity categories for burned.
    burned = mtbs.get_burn_severity(burned)
    logger.info(
        f"Number of burned shots after filtering for severity: {len(burned)}")

    # Add years since fire
    burned["YSF"] = burned.absolute_time.dt.year - burned.fire_ig_date.dt.year
    unburned["YSF"] = -100
    # Get rid of points over 35 years as we only have a few of those, and
    # points where YSF = 0, since it's unclear whether GEDI shot was taken
    # before or after fire.
    burned = burned[(burned.YSF != 0) & (burned.YSF < 36)]

    df = pd.concat([burned, unburned])
    logger.info(f"Total number of shots after joining: {len(df)}")
    df = gedi_utils.add_YSF_categories(df, 5)

    # Join with GEDI columns.
    gedi = gedi_utils.get_gedi_shots(SIERRAS_GEDI_ALL_COLUMNS, overlay.INDEX)
    GEDI_COLUMNS = ["agbd", "cover", "fhd_normal", "pai", "pai_z", "rh_25",
                    "rh_50", "rh_70", "rh_98", "elevation_difference_tdx"]
    # cols_to_use = gedi.columns.difference(df.columns)
    df = df.join(gedi[GEDI_COLUMNS], how="left")

    # Join with terrain.
    terrain = load_pickle(overlay.TERRAIN)
    cols_to_use = terrain.columns.difference(df.columns)
    df = df.join(terrain[cols_to_use], how="left")

    # Filter on terrain.
    df = filter.exclude_steep_slopes(df).drop(
        columns=["elevation_difference_tdx"])

    df = df[COLUMNS]

    burned = df[df.YSF > 0]
    unburned = df[df.YSF < 0]

    save_pickle(f"{OUTPUT_PATH}/burned.pkl", burned)
    save_pickle(f"{OUTPUT_PATH}/unburned.pkl", unburned)

    return burned, unburned


if __name__ == '__main__':
    run()
    filter_unburned_based_on_pre_fire_landcover()
