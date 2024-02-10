# Augments training or inference datasets with training features.
from pathlib import Path

import pandas as pd
from fastai.tabular.all import load_pickle
from src.data.processing import overlay


def MONTHLY_PREFIX(year):
    return f"monthly_landsat_overlay_{year}"


TTC_COLUMNS = ["tcc_2000", "tcc_2005", "tcc_2010", "tcc_2015"]


def add_landsat_monthly(
    df: pd.DataFrame,
    year: int
):
    all_overlays = list(Path(f"{overlay.OVERLAYS_PATH}").iterdir())
    monthly_overlays = [filename for filename in all_overlays
                        if filename.name.startswith(MONTHLY_PREFIX(year))]

    for filename in monthly_overlays:
        monthly_df = load_pickle(filename)
        cols = [col for col in monthly_df if (
            col.startswith("SR_") or col.startswith("NDVI"))]
        df_plus = df.join(monthly_df[cols], how="left")

    return df_plus


def add_tree_canopy_cover(
    df: pd.DataFrame
):
    tcc = load_pickle(overlay.TCC)
    df_plus = df.join(tcc[TTC_COLUMNS], how="left")
    return df_plus
