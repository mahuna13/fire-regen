import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pandas as pd
import numpy as np
import rioxarray as riox


BURN_DATA_RASTER = '/maps/fire-regen/data/rasters/burn_data_sierras.tif'
LAND_COVER_RASTER = '/maps/fire-regen/data/rasters/land_cover_sierras.tif'
TERRAIN_RASTER = '/maps/fire-regen/data/rasters/TERRAIN/terrain_stack.tif'


def LANDSAT_RASTER(year):
    return f"/maps/fire-regen/data/rasters/LANDSAT/{year}/out/landsat_{year}_stack.tif"


def DYNAMIC_WORLD_RASTER(year):
    return f"/maps/fire-regen/data/rasters/DYNAMIC_WORLD/dynamic_world_{year}.tif"


BURN_RASTER_BANDS = {0: 'burn_severity', 1: 'burn_year', 2: 'burn_counts'}
LAND_COVER_BANDS = {0: 'land_cover'}
TERRAIN_BANDS = {0: 'elevation', 1: 'slope', 2: 'aspect', 3: 'soil'}
LANDSAT_BANDS = {0: 'nbr', 1: 'ndvi', 2: 'SR_B1', 3: 'SR_B2',
                 4: 'SR_B3', 5: 'SR_B4', 6: 'SR_B5', 7: 'SR_B6', 8: 'SR_B7'}


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

    def sample_3x3(self, df: pd.DataFrame, x_coord: str, y_coord: str):
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
        for band_idx, band_name in self.bands.items():
            band_values = np.vstack(
                [self.raster.data[band_idx, ys[:, j], xs[:, i]]
                 for i in [0, 1, 2] for j in [0, 1, 2]]
            ).T

            df[f'{band_name}_3x3'] = list(band_values)
            df[f'{band_name}_mean'] = np.mean(band_values, axis=1)
            df[f'{band_name}_std'] = np.std(band_values, axis=1)
            df[f'{band_name}_median'] = np.median(band_values, axis=1)
        return df

    def sample(self, df: pd.DataFrame, x_coord: str, y_coord: str):
        xs = get_idx(self.raster.x.data, df[x_coord].values)
        ys = get_idx(self.raster.y.data, df[y_coord].values)

        # Calculate stats for each band. Attach to df.
        for band_idx, band_name in self.bands.items():
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
