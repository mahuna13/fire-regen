import ee
import shapely
from src.data.ee import ee_utils


def get_terrain_30m(polygon: shapely.Polygon, scale: int) -> ee.Image:
    region = ee_utils.gdf_to_ee_polygon(polygon)

    elevation = get_elevation_slope_and_aspect_30m()
    soil = get_soil()
    terrain_img = ee.Image.cat([elevation, soil]).clip(region)

    return terrain_img


def get_terrain_image(polygon: shapely.Polygon, scale: int) -> ee.Image:
    region = ee_utils.gdf_to_ee_polygon(polygon)

    elevation = get_elevation_slope_and_aspect()
    soil = get_soil()
    terrain_img = ee.Image.cat([elevation, soil]).clip(region)

    return terrain_img


def get_elevation_slope_and_aspect() -> ee.Image:
    '''
    Fetches elevation and slope from SRTM dataset using Earth Engine.
    '''
    elevation = ee.Image("CGIAR/SRTM90_V4").select("elevation")
    slope = ee.Terrain.slope(elevation)
    aspect = ee.Terrain.aspect(elevation)
    return ee.Image.cat([elevation, slope, aspect])


def get_elevation_slope_and_aspect_30m() -> ee.Image:
    '''
    Fetches 30m elevation and slope from SRTM dataset using Earth Engine.
    '''
    elevation = ee.Image("USGS/SRTMGL1_003").select("elevation")
    slope = ee.Terrain.slope(elevation)
    aspect = ee.Terrain.aspect(elevation)
    return ee.Image.cat([elevation, slope, aspect])


def get_soil() -> ee.Image:
    return ee.Image("CSP/ERGo/1_0/US/lithology").select(["b1"], ["soil"])
