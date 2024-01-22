import pandas as pd
from fastai.tabular.all import load_pickle, save_pickle
from src.constants import INTERMEDIATE_RESULTS
from src.data.pipelines.extract_gedi_data import SIERRAS_GEDI_ALL_COLUMNS
from src.data.utils import gedi_utils

OVERLAYS_PATH = f"{INTERMEDIATE_RESULTS}/overlays"
PIPELINES_PATH = f"{INTERMEDIATE_RESULTS}/pipelines"


def get_overlays_path(file_name: str):
    return f"{OVERLAYS_PATH}/{file_name}"


def get_pipelines_path(file_name: str):
    return f"{PIPELINES_PATH}/{file_name}"


'''
We want to do aggregate analysis - compare absolute metrics with respect of
years since fire with those same metrics in unburned areas.

Steps:
1. Separate regions on burned and unburned. For burned, have
   "years since fire" property.
2. Filter data. This can include many steps.
   a) For unburned - look at only those that are "trees". For burned, only
   those that were "trees" before they burned.
   b) TODO: pick other filtering steps
3. "Years since fire" should also be bucketed in categories. We should have
   "unburned", and then 5 year increments categories.
4. Choose metrics to evaluate and join them with other data. Potential
   metrics:
         * NDVI
         * NBR
         * AGBD
         * RH 70
         * RH 98
         * RH 50
         * RH 25
         * PAI
         * PAI - understory vs upper
         * CC
         * FHD
'''


def run(severity_analysis):
    if severity_analysis:
        burned = load_pickle(get_pipelines_path(
            "severity_burned_once_lc.pkl"))
    else:
        burned = load_pickle(get_pipelines_path("burned_once_lc.pkl"))
    unburned = load_pickle(get_pipelines_path("unburned_lc.pkl"))

    # Add years since fire
    burned["YSF"] = burned.absolute_time.dt.year - burned.fire_ig_date.dt.year
    unburned["YSF"] = -100

    # Step 3.
    df = pd.concat([burned, unburned])
    df = gedi_utils.add_YSF_categories(df, 5)

    # Step 4.
    # - load all of GEDI data.
    gedi = gedi_utils.get_gedi_shots(SIERRAS_GEDI_ALL_COLUMNS)
    GEDI_COLUMNS = ["agbd", "cover", "fhd_normal", "pai", "pai_z", "rh_25",
                    "rh_50", "rh_70", "rh_98", "elevation_difference_tdx"]
    # cols_to_use = gedi.columns.difference(df.columns)
    df = df.join(gedi[GEDI_COLUMNS], how="left")

    # Finally, join with terrain.
    terrain = load_pickle(get_overlays_path("terrain_overlay.pkl"))
    cols_to_use = terrain.columns.difference(df.columns)
    df = df.join(terrain[cols_to_use], how="left")

    # Filter on terrain.
    df = df[(df.slope < 30) & (df.elevation_difference_tdx.abs() < 50)
            ].drop(columns=["elevation_difference_tdx"])

    # And NDVI.
    ndvi = load_pickle(get_overlays_path("ndvi_overlay.pkl"))
    df = df.join(ndvi, how="left")

    if severity_analysis:
        save_pickle(get_pipelines_path(
            "severity_aggregated_info.pkl"), df)
    else:
        save_pickle(get_pipelines_path("aggregated_info.pkl"), df)

    return df


if __name__ == '__main__':
    run(severity_analysis=True)
