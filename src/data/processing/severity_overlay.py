import os
import re

import pandas as pd
from fastai.tabular.all import load_pickle, save_pickle
from src.data.adapters import mtbs
from src.data.processing import gedi_raster_matching
from src.data.utils import gedi_utils, raster
from src.utils.logging_util import get_logger

logger = get_logger(__file__)


GEDI_IDS_TO_REMOVE = [22791100300215022]


def get_gedi_shots(input_path: str):
    gedi_shots = gedi_utils.get_gedi_shots(input_path)
    if gedi_shots.index.isin(GEDI_IDS_TO_REMOVE).any():
        return gedi_shots.drop(GEDI_IDS_TO_REMOVE)
    else:
        return gedi_shots


def overlay_with_mtbs_fire_and_save(
        input_path: str,
        output_path: str,
        distance: int,
        post_fire_only: bool = True):
    overlay = overlay_with_mtbs_fires(input_path, distance, post_fire_only)
    return _save_overlay(overlay, output_path)


def overlay_with_mtbs_fires(
        input_path: str,
        distance: int,
        post_fire_only: bool = True):
    gedi_shots = get_gedi_shots(input_path)

    mtbs_fires = load_pickle(mtbs.MTBS_PERIMETERS_TRIMMED(distance))

    intersection = gedi_shots.sjoin(mtbs_fires,
                                    how="left",
                                    predicate="within")

    burned_shots = intersection[intersection.index_right.notna()]
    unburned_shots = intersection[intersection.index_right.isna()]

    if post_fire_only:
        # Look only at gedi shots post fire, not pre fire (relevant for the
        # most recent fires 2019-2022 that overlap with the dates GEDI was
        # sampled at).
        delta_time = burned_shots.absolute_time - pd.to_datetime(
            burned_shots.Ig_Date,
            utc=True,
            format='mixed')
        burned_shots["days_since_fire"] = delta_time.dt.days
        burned_shots = burned_shots[burned_shots.days_since_fire >= 0]

    # Assign total number of all fires for each GEDI shot.
    fire_count_col = "fire_count"
    unburned_shots[fire_count_col] = 0
    burned_shots[fire_count_col] = burned_shots.groupby(
        gedi_utils.INDEX).index_right.count()

    # For each burned shot, only keep the most recent fire details.
    most_recent_fire_idx = burned_shots.groupby(
        gedi_utils.INDEX).Ig_Date.transform(max) == burned_shots.Ig_Date
    most_recent_fires = burned_shots[most_recent_fire_idx]

    overlay = pd.concat([unburned_shots, most_recent_fires])
    overlay.drop(columns=["index_right"], inplace=True)

    return overlay


def overlay_with_mtbs_dnbr(
        input_path: str,
        output_path: str,
        distance: int,
        post_fire: bool = True):
    fire_occurrence_df = overlay_with_mtbs_fires(
        input_path, distance, post_fire)

    burned = fire_occurrence_df[fire_occurrence_df.fire_count > 0]

    for fire_id in burned.Event_ID.unique():
        print(fire_id)
        gedi_within = burned[burned.Event_ID == fire_id]
        fire_year = gedi_within.sample().Ig_Year.iloc[0]
        # return gedi_within
        dir_path = \
            f"{mtbs.MTBS_INDIVIDUAL_FIRES}/{int(fire_year)}/{fire_id.lower()}"

        if (os.path.exists(dir_path)):
            # MTBS for this fire exists.
            mtbs_files = os.listdir(dir_path)
            r = re.compile(f"^{fire_id.lower()}.*_dnbr\.tif")  # noqa: W605
            matches = list(filter(r.match, mtbs_files))

            if len(matches) == 1:
                print('file found')
                dnbr_file = matches[0]
                file_path = f"{dir_path}/{dnbr_file}"

                raster.reproject_raster(file_path, file_path)
                dnbr_raster = raster.RasterSampler(file_path, ["dnbr"])
                matched = gedi_raster_matching.sample_raster(
                    dnbr_raster, gedi_within, 2)
                print('rasters matched!')
                print(matched.columns)

                fire_occurrence_df.loc[matched.index,
                                       "dnbr_mean"] = matched.dnbr_mean
                fire_occurrence_df.loc[matched.index,
                                       "dnbr_std"] = matched.dnbr_std
                fire_occurrence_df.loc[matched.index,
                                       "dnbr_median"] = matched.dnbr_median
            else:
                logger.warn(
                    f"Found zero or more than one dnbr \
                        file matching the query in the directory {dir_path}.")
        else:
            logger.warn(f"Cannot find directory for path {dir_path}.")

    return _save_overlay(fire_occurrence_df, output_path)


def overlay_with_mtbs_severity_categories(input_path: str, output_path: str):
    gedi_shots = get_gedi_shots(input_path)

    overlay = gedi_raster_matching.match_burn_raster(gedi_shots, kernel=2)
    return _save_overlay(overlay, output_path)


def _save_overlay(df: pd.DataFrame, output_path: str):
    df = df.drop(columns=["longitude", "latitude", "geometry"])

    if output_path is None:
        return df

    logger.info("Saving the results.")
    save_pickle(output_path, df)
    return df
