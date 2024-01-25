import os

import numpy as np
import pandas as pd
import rasterio as rio
import rioxarray as riox
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject
from src.utils.logging_util import get_logger

logger = get_logger(__file__)

pd.options.mode.chained_assignment = None  # default='warn'


class RasterSampler:
    def __init__(
            self,
            raster_file_path: str,
            bands: list[str] = None,
            bands_map: dict = None):
        self.raster = riox.open_rasterio(raster_file_path)
        # Need to provide at least one - bands or bands_map.
        if bands is None and bands_map is None:
            raise Exception("bands or bands_map argument must be provided.")

        if bands_map is not None:
            self.bands_map = bands_map
            return

        self.bands_map = dict(zip(range(len(bands)), bands))

    def sample_2x2(self, df: pd.DataFrame, x_coord: str, y_coord: str,
                   expanded: bool = False):
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
        all_bands = []
        for band_idx, band_name in self.bands_map.items():
            band_values = np.vstack(
                [self.raster.data[band_idx, ys[:, j], xs[:, i]]
                 for i in [0, 1] for j in [0, 1]]
            ).T

            data = {
                f'{band_name}_mean': np.mean(band_values, axis=1),
                f'{band_name}_std': np.std(band_values, axis=1),
                f'{band_name}_median': np.median(band_values, axis=1)
            }

            if expanded:
                # Add additional columns
                data[f'{band_name}_2x2'] = list(band_values)
                data[f'{band_name}_min'] = np.min(band_values, axis=1)
                data[f'{band_name}_max'] = np.max(band_values, axis=1)

            new_df = pd.DataFrame(index=df.index, data=data)
            all_bands.append(new_df)

        return pd.concat([df] + all_bands, axis=1)

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
        for band_idx, band_name in self.bands_map.items():
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
        for band_idx, band_name in self.bands_map.items():
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


def merge_raster_tiles(path, output_file_path):
    if os.path.exists(output_file_path):
        # We've merged the tiles already, early exit.
        return

    raster_files = list(path.iterdir())

    logger.debug('Load tif tiles')
    raster_to_mosaic = []
    for tif in raster_files:
        raster = rio.open(tif)
        raster_to_mosaic.append(raster)

    logger.debug('Merge rasters.')
    mosaic, output = merge(raster_to_mosaic)

    logger.debug('Write output')
    output_meta = raster.meta.copy()
    output_meta.update(
        {"driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": output,
         }
    )
    with rio.open(output_file_path, "w", **output_meta) as m:
        m.write(mosaic)
