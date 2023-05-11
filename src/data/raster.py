import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pandas as pd
import numpy as np
import xarray
import rioxarray as riox


BURN_DATA_RASTER = '/maps/fire-regen/data/rasters/burn_data_sierras.tif'
LAND_COVER_RASTER = '/maps/fire-regen/data/rasters/land_cover_sierras.tif'

BURN_RASTER_BANDS = {0: 'burn_severity', 1: 'burn_year', 2: 'burn_counts'}
LAND_COVER_BANDS = {0: 'land_cover'}


class Raster:
    def __init__(self, raster_file_path: str, bands: dict[str, int]):
        self.raster = self.read_raster(raster_file_path)
        self.bands = bands

    def read_raster(self, directory: str) -> rio.DatasetReader:
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

    def get_band_index(self, band_name: str) -> int:
        return self.bands[band_name]

    def transform_geo_to_xy_coords(self, geo_coords: list[tuple]):
        transformer = rio.transform.AffineTransformer(self.raster.transform)
        return [transformer.rowcol(x[0], x[1]) for x in geo_coords]

    def transform_xy_to_geo_coords(self, xy_coords: list[tuple]):
        transformer = rio.transform.AffineTransformer(self.raster.transform)
        return [transformer.xy(x[0], x[1]) for x in xy_coords]

    def sample(self, xy_coords: list[tuple]):
        return self.raster.sample(xy_coords)


class RasterSampler:
    def __init__(self, raster_file_path: str, bands: dict[str, int]):
        self.raster = riox.open_rasterio(raster_file_path)
        self.bands = bands

    def sample_2x2(self, df: pd.DataFrame, x_coord: str, y_coord: str):
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
        for band_idx, band_name in self.bands.items():
            band_values = np.vstack(
                [self.raster.data[band_idx, ys[:, j], xs[:, i]]
                 for i in [0, 1] for j in [0, 1]]
            ).T

            df[f'{band_name}_2x2'] = list(band_values)
            df[f'{band_name}_mean'] = np.mean(band_values, axis=1)
            df[f'{band_name}_std'] = np.std(band_values, axis=1)
            df[f'{band_name}_median'] = np.median(band_values, axis=1)
        return df


class RasterSamplerOld:
    def __init__(self, raster: Raster):
        self.raster = raster

    def sample(self, df_input: pd.DataFrame, x_coord: str, y_coord: str,
               kernel_size: int, bands: list[str]) -> pd.DataFrame:
        df = df_input.copy()

        # Calculate coordinates for sampling the raster.
        geo_coords = [(x, y) for x, y in zip(df[x_coord], df[y_coord])]
        geo_coord_with_kernel = self._get_geo_coords_with_kernel(geo_coords,
                                                                 kernel_size)

        # Sample raster.
        flattened_samples = np.array(
            [x for x in self.raster.sample(geo_coord_with_kernel)])

        # Calculate stats for each band. Attach to df.
        for band_index, band_name in bands:
            band_index = self.raster.get_band_index(band_name)
            band_flattened = flattened_samples[:, band_index]

            band_values = band_flattened.reshape(
                len(geo_coords), kernel_size ** 2)

            df[f'{band_name}_3x3'] = list(band_values)
            df[f'{band_name}_sample'] = band_values[:, kernel_size + 1]
            df[f'{band_name}_mean'] = np.mean(band_values, axis=1)
            df[f'{band_name}_std'] = np.std(band_values, axis=1)
            df[f'{band_name}_median'] = np.median(band_values, axis=1)
        return df

    def sample_2x2(self, df: pd.DataFrame, x_coord: str, y_coord: str,
                   bands: list[str]):
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
        for band_idx, band_name in bands.items():
            band_values = np.vstack(
                [self.raster.data[band_idx, ys[:, j], xs[:, i]]
                 for i in [0, 1] for j in [0, 1]]
            ).T

            df[f'{band_name}_3x3'] = list(band_values)
            df[f'{band_name}_mean'] = np.mean(band_values, axis=1)
            df[f'{band_name}_std'] = np.std(band_values, axis=1)
            df[f'{band_name}_median'] = np.median(band_values, axis=1)
        return df

    def _get_geo_coords_with_kernel(self, geo_coords: list[tuple], kernel: int):
        xy_coord_list = self.raster.transform_geo_to_xy_coords(geo_coords)

        xy_coord_with_kernel = [
            x for y in xy_coord_list
            for x in self._create_kernel_coords(y, kernel)]

        return self.raster.transform_xy_to_geo_coords(xy_coord_with_kernel)

    def _create_kernel_coords(self, coords: tuple, kernel_size: int):
        x, y = coords
        min_x, min_y = x - kernel_size // 2, y - kernel_size // 2
        kernel_coords = []

        for dy in range(0, kernel_size):
            for dx in range(0, kernel_size):
                kernel_coords.append((min_x + dx, min_y + dy))

        return kernel_coords


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
