import numpy as np
import pandas as pd
import rasterio as rio
import rioxarray as riox
from rasterio.warp import Resampling, calculate_default_transform, reproject

pd.options.mode.chained_assignment = None  # default='warn'


class RasterSampler:
    def __init__(self, raster_file_path: str, bands: list[str]):
        self.raster = riox.open_rasterio(raster_file_path)
        self.bands = bands

    def sample_2x2(self, df: pd.DataFrame, x_coord: str, y_coord: str,
                   debug: bool = False):
        xs = get_idxs_two_nearest(self.raster.x.data, df[x_coord].values)
        ys = get_idxs_two_nearest(self.raster.y.data, df[y_coord].values)
        valid = np.logical_and.reduce(
            (
                (np.min(xs, axis=1) >= 0),
                (np.max(xs, axis=1) < self.raster.shape[2]),
                (np.min(ys, axis=1) >= 0),
                (np.max(ys, axis=1) < self.raster.shape[1]),
            )
        )
        xs = xs[valid]
        ys = ys[valid]
        df = df.loc[valid]

        # Calculate stats for each band. Attach to df.
        for band_idx in range(len(self.bands)):
            band_name = self.bands[band_idx]
            band_values = np.vstack(
                [self.raster.data[band_idx, ys[:, j], xs[:, i]]
                 for i in [0, 1] for j in [0, 1]]
            ).T

            if debug:
                # Could be helpful to get the values from all 4 cells.
                df[f'{band_name}_2x2'] = list(band_values)

            df[f'{band_name}_mean'] = np.mean(band_values, axis=1)
            df[f'{band_name}_std'] = np.std(band_values, axis=1)
            df[f'{band_name}_median'] = np.median(band_values, axis=1)
        return df

    def sample_3x3(self, df: pd.DataFrame, x_coord: str, y_coord: str,
                   debug: bool = False):
        xs = get_idxs_three_nearest(self.raster.x.data, df[x_coord].values)
        ys = get_idxs_three_nearest(self.raster.y.data, df[y_coord].values)
        valid = np.logical_and.reduce(
            (
                (np.min(xs, axis=1) >= 0),
                (np.max(xs, axis=1) < self.raster.shape[2]),
                (np.min(ys, axis=1) >= 0),
                (np.max(ys, axis=1) < self.raster.shape[1]),
            )
        )
        xs = xs[valid]
        ys = ys[valid]
        df = df.loc[valid]

        # Calculate stats for each band. Attach to df.
        for band_idx in range(len(self.bands)):
            band_name = self.bands[band_idx]
            band_values = np.vstack(
                [self.raster.data[band_idx, ys[:, j], xs[:, i]]
                 for i in [0, 1, 2] for j in [0, 1, 2]]
            ).T

            if debug:
                # Could be helpful to get the values from all 4 cells.
                df[f'{band_name}_3x3'] = list(band_values)
            df[f'{band_name}_mean'] = np.mean(band_values, axis=1)
            df[f'{band_name}_std'] = np.std(band_values, axis=1)
            df[f'{band_name}_median'] = np.median(band_values, axis=1)
        return df

    def sample(self, df: pd.DataFrame, x_coord: str, y_coord: str):
        xs = get_idx(self.raster.x.data, df[x_coord].values)
        ys = get_idx(self.raster.y.data, df[y_coord].values)

        # Calculate stats for each band. Attach to df.
        for band_idx in range(len(self.bands)):
            band_name = self.bands[band_idx]
            band_values = self.raster.data[band_idx, ys, xs]
            df[f'{band_name}'] = list(band_values)
        return df


def get_idxs_two_nearest(array, values):
    """Find the 2x2 pixel box in a raster that best covers a small
     circle around the given coords.

    If the coords fall outside the raster, the nearest pixel is
    the border pixel, but the second-nearest pixel will be listed
    as an out-of-bounds index.

    Copied from https://github.com/ameliaholcomb/biomass-degradation.

    Args:
        array (n,): array of of raster coordinate values
                    e.g. from raster.x.data
        values (k,): coordinate values to find in the raster
    Returns:
        np.array(k,): indices containing coordinate values
    """
    half_pixel = (array[1] - array[0]) / 2
    array_center = array + half_pixel
    argmins = np.zeros((*values.shape, 2), dtype=np.int64)
    for i, value in enumerate(values):
        argmins[i, 0] = (np.abs(array_center - value)).argmin()
        if value < array_center[argmins[i, 0]]:
            argmins[i, 1] = argmins[i, 0] - 1
        else:
            argmins[i, 1] = argmins[i, 0] + 1

    return argmins


def get_idxs_three_nearest(array, values):
    """Find the 3x3 pixel box in a raster that best covers a small
     circle around the given coords.

    If the coords fall outside the raster, the nearest pixel is
    the border pixel, but the second-nearest pixel will be listed
    as an out-of-bounds index.

    Copied from https://github.com/ameliaholcomb/biomass-degradation.

    Args:
        array (n,): array of of raster coordinate values
                    e.g. from raster.x.data
        values (k,): coordinate values to find in the raster
    Returns:
        np.array(k,): indices containing coordinate values
    """
    argmins = np.zeros((*values.shape, 3), dtype=np.int64)
    for i, value in enumerate(values):
        argmin = (np.abs(array - value)).argmin()
        argmins[i, 0] = argmin - 1
        argmins[i, 1] = argmin
        argmins[i, 2] = argmin + 1

    return argmins


def get_idx(array, values):
    """Find the pixel index in a raster covering the given coords.

    Copied from https://github.com/ameliaholcomb/biomass-degradation.

    Args:
        array (n,): array of of raster coordinate values
                    e.g. from raster.x.data
        values (k,): coordinate values to find in the raster
    Returns:
        np.array(k,): indices containing coordinate values
    """
    nearest_idxs = np.zeros_like(values, dtype=np.int64)
    for i, value in enumerate(values):
        nearest_idxs[i] = (np.abs(array - value)).argmin()
    return nearest_idxs


def reproject_raster(file_path: str, out_file_path: str,
                     dst_crs: str = 'EPSG:4326'):

    with rio.open(file_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        with rio.open(out_file_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
