import ee
import geemap
import geemap.colormaps as cm
import numpy as np
import shapely

BURN_SEVERITY_PALETTE = ['000000', '006400',
                         '7FFFD4', 'FFFF00', 'FF0000', '7FFF00', 'FFFFFF']
BURN_COUNT_PALETTE = ['FFFFFF', 'fecc5c', 'fd8d3c', 'f03b20', 'bd0026']
BURN_SEVERITY_MIN = 0
BURN_SEVERITY_MAX = 6
LAND_COVER_MAP = 'NLCD 2019 CONUS Land Cover'
GRAY_MAP = 'Esri.WorldGrayCanvas'
TOPO_MAP = 'USGS.USTopo'


def viz_burn_severity(img: ee.Image, polygon: shapely.Polygon,
                      bands: list[str],
                      band_names: list[str] = None,
                      map: geemap.Map = None,
                      landcover: bool = False,
                      scale: int = 30,
                      legend: bool = True) -> geemap.Map:
    '''
    Plots image on a map clipped to a specified polygon.

    Each band is added as a separate layer on the map.

    '''
    region = gdf_to_ee_polygon(polygon)
    img_stats = geemap.image_stats(img.clip(region), scale=scale).getInfo()

    if band_names is None:
        band_names = bands

    if map is None:
        if landcover:
            basemap = LAND_COVER_MAP
        else:
            basemap = TOPO_MAP
        map = geemap.Map(center=(polygon.centroid.y, polygon.centroid.x),
                         zoom=9, basemap=basemap, height=1500)

        # Plot the perimeter of area of interest.
    perimeter_vis = {'color': 'ff5722', 'fillColor': 'ff8a50', 'width': 2,
                     'opacity': 0.5}
    map.addLayer(region, perimeter_vis, 'Region of interest')

    for band, name in zip(bands, band_names):
        print((band, name))
        if band == 'Severity':
            band_vis = {'bands': band, 'palette': BURN_SEVERITY_PALETTE,
                        'min': BURN_SEVERITY_MIN, 'max': BURN_SEVERITY_MAX}
            map.addLayer(img.clip(region), band_vis, name)
            if legend:
                print('adding legeeend')
                map.add_colorbar(band_vis, label=name, layer_name=band)
        else:
            band_min = img_stats['min'][band]
            band_max = img_stats['max'][band]
            band_vis = {'bands': band, 'palette': cm.palettes.inferno_r,
                        'min': band_min, 'max': band_max}
            map.addLayer(img.clip(region), band_vis, name)
            if legend:
                map.add_colorbar(band_vis, label=name, layer_name=band)

    return map


def viz_burn_counts(img: ee.Image, polygon: shapely.Polygon,
                    bands: list[str],
                    band_names: list[str] = None,
                    map: geemap.Map = None,
                    landcover: bool = False,
                    scale: int = 30) -> geemap.Map:
    '''
    Plots image on a map clipped to a specified polygon.

    Each band is added as a separate layer on the map.

    '''
    region = gdf_to_ee_polygon(polygon)
    img_stats = geemap.image_stats(img.clip(region), scale=scale).getInfo()

    if band_names is None:
        band_names = bands

    if map is None:
        if landcover:
            basemap = LAND_COVER_MAP
        else:
            basemap = GRAY_MAP
        map = geemap.Map(center=(polygon.centroid.y, polygon.centroid.x),
                         zoom=9, basemap=basemap, height=1500)

        # Plot the perimeter of area of interest.
    perimeter_vis = {'color': 'ff5722', 'fillColor': 'ff8a50', 'width': 2,
                     'opacity': 0.5}
    map.addLayer(region, perimeter_vis, 'Region of interest')

    for band, name in zip(bands, band_names):
        band_min = img_stats['min'][band]
        band_max = img_stats['max'][band]
        band_vis = {'bands': band, 'palette': cm.palettes.inferno_r,
                    'min': band_min, 'max': band_max}
        map.addLayer(img.clip(region), band_vis, name)
        map.add_colorbar(band_vis, label=name, layer_name=band)

    return map


def viz_single_region(img: ee.Image, polygon: shapely.Polygon,
                      bands: list[str],
                      band_names: list[str] = None,
                      palette: cm.Box = cm.palettes.inferno_r,
                      scale: int = 5000,
                      map: geemap.Map = None) -> geemap.Map:
    '''
    Plots image on a map clipped to a specified polygon.

    Each band is added as a separate layer on the map.

    '''
    region = gdf_to_ee_polygon(polygon)
    img_stats = geemap.image_stats(img.clip(region), scale=scale).getInfo()

    if band_names is None:
        band_names = bands

    if map is None:
        map = geemap.Map(center=(polygon.centroid.y, polygon.centroid.x),
                         zoom=9)

    for band, name in zip(bands, band_names):
        band_min = img_stats['min'][band]
        band_max = img_stats['max'][band]
        band_vis = {'bands': band, 'palette': palette,
                    'min': band_min, 'max': band_max}
        map.addLayer(img.clip(region), band_vis, name)
        map.add_colorbar(band_vis, label=name, layer_name=band)

    return map


def viz_image(img: ee.Image, polygon: shapely.Polygon,
              bands: list[str],
              band_names: list[str] = None,
              palette: cm.Box = cm.palettes.inferno_r,
              scale: int = 5000,
              map: geemap.Map = None) -> geemap.Map:
    '''
    Plots image on a map clipped to a specified polygon.

    Each band is added as a separate layer on the map.

    '''

    if band_names is None:
        band_names = bands

    if map is None:
        map = geemap.Map(center=(polygon.centroid.y, polygon.centroid.x),
                         zoom=9)

    for band, name in zip(bands, band_names):
        band_vis = {'bands': band, 'palette': palette}
        map.addLayer(img, band_vis, name)
        map.add_colorbar(band_vis, label=name, layer_name=band)

    return map


def gdf_to_ee_polygon(gdf_polygon: shapely.Polygon):
    ''' Helper to convert GeoPandas geometry to Earth Engine geometry. '''
    x, y = gdf_polygon.exterior.coords.xy
    coords = np.dstack((x, y)).tolist()
    return ee.Geometry.Polygon(coords)
