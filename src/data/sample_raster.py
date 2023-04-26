import rasterio as rio
import pandas as pd
import numpy as np

RASTER_FILE = '/maps/fire-regen/data/rasters/burn_mosaic.tif'
BURN_DATA_RASTER_FILE = '/maps/fire-regen/data/rasters/burn_data_sierras.tif'


def create_kernel_coords(coords: tuple, kernel_size: int):
    x, y = coords
    min_x, min_y = x - kernel_size // 2, y - kernel_size // 2
    kernel_coords = []

    for dy in range(0, kernel_size):
        for dx in range(0, kernel_size):
            kernel_coords.append((min_x + dx, min_y + dy))

    return kernel_coords

# TODO: Make the below function more general to any bands.


def sample_raster(gedi_df: pd.DataFrame, kernel_size: int):
    raster_obj = read_raster(BURN_DATA_RASTER_FILE)
    geo_coord_list = [(x, y) for x, y in zip(
        gedi_df.lon_lowestmode, gedi_df.lat_lowestmode)]

    transformer = rio.transform.AffineTransformer(raster_obj.transform)
    xy_coord_list = [transformer.rowcol(x[0], x[1]) for x in geo_coord_list]

    xy_coord_with_kernel = [
        x for y in xy_coord_list for x in create_kernel_coords(y, kernel_size)]

    geo_coord_with_kernel = [transformer.xy(
        x[0], x[1]) for x in xy_coord_with_kernel]

    flattened_samples = np.array(
        [x for x in raster_obj.sample(geo_coord_with_kernel)])

    severity_flattened = flattened_samples[:, 0]
    year_flattened = flattened_samples[:, 1]
    burn_counts_flattened = flattened_samples[:, 2]

    severity = severity_flattened.reshape(
        len(geo_coord_list), kernel_size ** 2)
    year = year_flattened.reshape(len(geo_coord_list), kernel_size ** 2)
    burn_counts = burn_counts_flattened.reshape(
        len(geo_coord_list), kernel_size ** 2)

    severity_samples = severity[:, kernel_size + 1]
    year_samples = year[:, kernel_size + 1]
    burn_counts_samples = burn_counts[:, kernel_size + 1]

    severity_mean = np.mean(severity, axis=1)
    year_mean = np.mean(year, axis=1)

    severity_std = np.std(severity, axis=1)
    year_std = np.std(year, axis=1)

    severity_median = np.median(severity, axis=1)
    burn_counts_mean = np.mean(burn_counts, axis=1)

    burn_counts_std = np.std(burn_counts, axis=1)

    burn_counts_median = np.median(burn_counts, axis=1)

    gedi_df['burn_severity_3x3'] = list(severity)
    gedi_df['burn_year_3x3'] = list(year)
    gedi_df['burn_severity_sample'] = severity_samples
    gedi_df['burn_year_sample'] = year_samples
    gedi_df['burn_severity_mean'] = severity_mean
    gedi_df['burn_year_mean'] = year_mean
    gedi_df['burn_severity_std'] = severity_std
    gedi_df['burn_year_std'] = year_std
    gedi_df['burn_severity_median'] = severity_median
    gedi_df['burn_counts_3x3'] = list(burn_counts)
    gedi_df['burn_counts_sample'] = burn_counts_samples
    gedi_df['burn_counts_mean'] = burn_counts_mean
    gedi_df['burn_counts_std'] = burn_counts_std
    gedi_df['burn_counts_median'] = burn_counts_median
    return gedi_df


# Old method - not used. TODO: evaluate if it should be deleted.
def sample_burn_raster(gedi_df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    raster_obj = read_raster(RASTER_FILE)
    coords = np.array(
        list(zip(gedi_df['lon_lowestmode'], gedi_df['lat_lowestmode'])))

    indexes = np.apply_along_axis(
        lambda x: return_index_col(x, raster_obj), 1, coords)

    severity_windows = np.apply_along_axis(
        lambda idx: retrieve_window_array(idx,
                                          raster_obj,
                                          window_size,
                                          1), 1, indexes)

    severity_samples = severity_windows[:, window_size + 1]
    year_windows = np.apply_along_axis(
        lambda idx: retrieve_window_array(idx,
                                          raster_obj,
                                          window_size,
                                          2), 1, indexes)
    year_samples = year_windows[:, window_size + 1]
    gedi_df['burn_severity_3x3'] = pd.Series(list(severity_windows))
    gedi_df['burn_year_3x3'] = pd.Series(list(year_windows))
    gedi_df['burn_severity_sample'] = severity_samples
    gedi_df['burn_year_sample'] = year_samples
    return gedi_df


def read_raster(directory: str) -> rio.DatasetReader:
    '''
    Open raster as array file and evaluate if is in right projection
    compared with GEDI data (ESPG:4326).
    '''
    raster = rio.open(directory)
    crs = str(raster.crs)
    if crs == 'EPSG:4326':
        return raster
    else:
        raise Warning("Your raster file is not in crs 'EPSG:4326'")


def return_index_col(coords: tuple, raster_obj: rio.DatasetReader) \
        -> np.ndarray:
    """
    Return the row and column from a raster based on a lat and long
    of a point stored in a tuple.
    """
    row_index, column_index = raster_obj.index(coords[0], coords[1])
    row_col_array = np.array([row_index, column_index])
    return row_col_array


def retrieve_window_array(idx: np.ndarray,
                          raster_obj: rio.DatasetReader,
                          window_size: int,
                          band_index: int) -> np.ndarray:
    """
    Retrieves a window from a raster centered on the row and column idx.
    """
    raster = raster_obj.read(band_index)
    window = raster[idx[0] - (window_size//2): idx[0] + (window_size//2 + 1),
                    idx[1] - (window_size//2): idx[1] + (window_size//2 + 1)]
    print(window)
    return window.flatten()
