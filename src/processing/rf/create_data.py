from src.data import gedi_pipeline
from src.data import gedi_raster_matching
from src.utils.logging_util import get_logger
from src.constants import DATA_PATH
from fastai.tabular.all import save_pickle

logger = get_logger(__file__)


def create_and_save_data_for_rf(gedi, start_year, end_year, save: bool = False, save_folder: str = None):
    # Match with 5 years of Landsat data from the past.
    gedi_match = gedi.copy()
    kernel = 3

    counter = 0
    for year in range(start_year, end_year):
        logger.debug(f'Matching gedi shots from {year}')
        raster = gedi_raster_matching.get_landsat_raster_sampler(year)

        logger.debug(f"Sampling raster.")
        gedi_match = gedi_raster_matching.sample_raster(
            raster, gedi_match, kernel=kernel)

        print(f"Process columns.")
        gedi_match = process_spectral_column_names(gedi_match, year, kernel)
        counter += 1

        # First, save the picke file.
        if save:
            logger.debug(
                f"Save DF in a pickle file. Training data for year {year + 1}")
            save_pickle(
                f"{save_folder}/gedi_match_{year + 1}.pkl", gedi_match)

        if counter >= 5:
            # Drop the oldest year columns.
            logger.debug(f"Dropping columns from {year-4}")
            spectral = gedi_raster_matching.get_landsat_bands(year-4)
            gedi_match = gedi_match.drop(
                columns=[f"{x}_{year-4}" for x in spectral])

    return gedi_match


def process_spectral_column_names(df, year, kernel):
    spectral = gedi_raster_matching.get_landsat_bands(year)
    df = df.rename(columns=dict(
        zip([f"{x}_mean" for x in spectral], [f"{x}_{year}" for x in spectral])))
    df = df.drop(columns=[f"{x}_{kernel}x{kernel}" for x in spectral] +
                 [f"{x}_median" for x in spectral] + [f"{x}_std" for x in spectral])
    return df