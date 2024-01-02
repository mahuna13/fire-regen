from src.constants import GEDI_INTERMEDIATE_PATH
from src.utils.logging_util import get_logger
from fastai.tabular.all import load_pickle, save_pickle

logger = get_logger(__file__)

OVERLAYS_PATH = f"{GEDI_INTERMEDIATE_PATH}/overlays"


def get_overlay(file_name: str):
    return load_pickle(f"{OVERLAYS_PATH}/{file_name}")


if __name__ == '__main__':
    # Load all GEDI shots overlayed with MTBS data.
    shots = get_overlay("mtbs_severity_overlay.pkl")
    logger.info(f"Total number of GEDI shots to process: {len(shots)}")

    # Get rid of all the shots that fell around the boundary of any fire.
    boundary_shots = get_overlay("burn_boundary_overlay.pkl")
    shots = shots[~shots.index.isin(boundary_shots.index)]
    logger.info(f"Number of GEDI shots after filtering for areas around \
        fire boundaries: {len(shots)}")

    # Get fires from call fires.
    all_fires = get_overlay("all_fires_overlay.pkl")

    # Eliminate all shots that fall within small and recent fires.
    small_recent = all_fires[(all_fires.Shape_Area_Acres < 1000)
                             & ((all_fires.YEAR_ >= 1984))]
    shots = shots[~shots.index.isin(small_recent.index)]
    logger.info(
        f"Number of GEDI shots after filtering small fires: {len(shots)}")

    # Eliminate all shots that burned 10 years before 1984.
    old_fires = all_fires[(all_fires.YEAR_ < 1984) & (all_fires.YEAR_ >= 1974)]
    shots = shots[~shots.index.isin(old_fires.index)]
    logger.info(
        f"Number of GEDI shots after filtering old fires: {len(shots)}")

    # burned, burned once, and unburned - report length.
    burned_once = shots[shots.fire_count == 1]
    burned_multiple = shots[shots.fire_count > 1]
    unburned = shots[shots.fire_count == 0]
    logger.info(
        f"Number of GEDI shots that burned once: {len(burned_once)}")
    logger.info(
        f"Number of GEDI shots that burned many times: {len(burned_multiple)}")
    logger.info(
        f"Number of GEDI shots that didn't burn: {len(unburned)}")

    save_pickle(f"{OVERLAYS_PATH}/burned_once.pkl", burned_once)
    save_pickle(f"{OVERLAYS_PATH}/burned_multiple.pkl", burned_multiple)
    save_pickle(f"{OVERLAYS_PATH}/unburned.pkl", unburned)
