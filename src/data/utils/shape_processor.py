import geopandas as gpd

from shapely.geometry import box
from shapely import unary_union


def get_union(region_gpd: gpd.GeoDataFrame, crs: int = None) \
        -> gpd.GeoDataFrame:
    '''
    If Geo DataFrame contains many rows with many geometries, create a
    union of all of them, outputing another df with a single row and single
    geometry.
    '''
    if crs is None:
        crs = region_gpd.crs

    series = gpd.GeoSeries(
        [unary_union(region_gpd.to_crs(crs).geometry)]).set_crs(crs)
    return gpd.GeoDataFrame({'geometry': series})


def get_convex_hull(region_gpd: gpd.GeoDataFrame, crs: int = None) \
        -> gpd.GeoDataFrame:
    ''' Return a convex hull around a shape, as Geo DataFrame. '''
    return gpd.GeoDataFrame(
        {'geometry': get_union(region_gpd, crs).convex_hull})


def get_envelope(region_gpd: gpd.GeoDataFrame, crs: int = None) \
        -> gpd.GeoDataFrame:
    ''' Return an envelope around a shape, as Geo DataFrame. '''
    return gpd.GeoDataFrame({'geometry': get_union(region_gpd, crs).envelope})


def get_box(region_gpd: gpd.GeoDataFrame, crs: int = None) -> gpd.GeoDataFrame:
    ''' Return a bounding box around a shape, as Geo DataFrame. '''
    box_envelope = box(*(get_union(region_gpd, crs).geometry.iloc[0].bounds))
    if crs is None:
        crs = region_gpd.crs

    return gpd.GeoDataFrame({'geometry':
                             gpd.GeoSeries([box_envelope]).set_crs(crs)})