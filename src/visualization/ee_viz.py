import ee
import geemap
import geemap.colormaps as cm
import shapely
import src.data.ee.ee_utils as ee_utils


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
    region = ee_utils.gdf_to_ee_polygon(polygon)
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
        band_min = img_stats['min'][band]
        band_max = img_stats['max'][band]
        band_vis = {'bands': band, 'palette': cm.palettes.inferno_r,
                    'min': band_min, 'max': band_max}
        map.addLayer(img.clip(region), band_vis, name)
        if legend:
            map.add_colorbar(band_vis, label=name, layer_name=band)

    return map
