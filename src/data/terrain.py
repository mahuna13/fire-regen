import ee
import shapely
import src.data.ee_utils as ee_utils


def get_terrain_image(polygon: shapely.Polygon, scale: int) -> ee.Image:
    region = ee_utils.gdf_to_ee_polygon(polygon)

    elevation = get_elevation_slope_and_aspect()
    soil = get_soil()
    terrain_img = ee.Image.cat([elevation, soil]).clip(region)

    return terrain_img

    # normalize image to values of 0-255, to cast to uint8 and save on memory.
    # terrain_norm = _normalize_image(terrain_img, region, scale)

    # return terrain_norm.cast({'elevation': 'uint8',
    #                          'slope': 'uint8',
    #                          'aspect': 'uint8',
    #                          'soil': 'uint8'})


def get_elevation_slope_and_aspect() -> ee.Image:
    '''
    Fetches elevation and slope from SRTM dataset using Earth Engine.
    '''
    elevation = ee.Image("CGIAR/SRTM90_V4").select("elevation")
    slope = ee.Terrain.slope(elevation)
    aspect = ee.Terrain.aspect(elevation)
    return ee.Image.cat([elevation, slope, aspect])


def get_soil() -> ee.Image:
    return ee.Image("CSP/ERGo/1_0/US/lithology").select(["b1"], ["soil"])


def _normalize_image(img: ee.Image, region: ee.Geometry.Polygon, scale: int):
    # calculate the min and max value of an image
    minMax = img.reduceRegion(reducer=ee.Reducer.minMax(),
                              geometry=region,
                              scale=scale,
                              maxPixels=10e9
                              )

    def _normalize_band(name):
        band_name = ee.String(name)
        band = img.select(band_name)
        return band.unitScale(
            ee.Number(minMax.get(band_name.cat('_min'))),
            ee.Number(minMax.get(band_name.cat('_max')))
        ).multiply(255)

    # use unit scale to normalize the pixel values
    return ee.ImageCollection.fromImages(
        img.bandNames().map(_normalize_band)
    ).toBands().rename(img.bandNames())
