import ee
from src.data.ee import ee_utils


'''
    A lot of this code is based on the paper for deriving Fire severity metrics
    from https://www.mdpi.com/2072-4292/10/6/879.
'''

# Enter beginning and end days for imagery season as julian dates
START_DAY = 152
END_DAY = 273
# In the paper, they used startday=91 and endday=181 for fires in Arizona,
# New Mexico, and Utah.
# We used startday=152 and endday=273 for fires in California, Montana,
# Washington, and Wyoming.


def get_NBR_index(ls_img: ee.Image):
    nbr = ls_img.normalizedDifference(['SR_B5', 'SR_B7'])
    return nbr.select([0], ['nbr']) \
        .copyProperties(ls_img, ['system:time_start'])


def get_NDVI_index(ls_img: ee.Image):
    ndvi = ls_img.normalizedDifference(['SR_B5', 'SR_B4'])
    return ndvi.select([0], ['ndvi']) \
        .copyProperties(ls_img, ['system:time_start'])


def mask_cloud_pixels_landsat_8(image: ee.Image):
    qaMask = image.select(['QA_PIXEL']).bitwiseAnd(int('11111', 2)).eq(0)
    saturationMask = image.select(['QA_RADSAT']).eq(0)

    opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBand = image.select('ST_B10').multiply(0.00341802).add(149.0)

    return image.addBands(opticalBands, None, True) \
        .addBands(thermalBand, None, True) \
        .updateMask(qaMask) \
        .updateMask(saturationMask)


def mask_cloud_pixels_landsat_5(image: ee.Image):
    qaMask = image.select(['QA_PIXEL']).bitwiseAnd(int('11111', 2)).eq(0)
    saturationMask = image.select(['QA_RADSAT']).eq(0)

    opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermalBand = image.select('ST_B6').multiply(0.00341802).add(149.0)

    return image.addBands(opticalBands, None, True) \
        .addBands(thermalBand, None, True) \
        .updateMask(qaMask) \
        .updateMask(saturationMask)


def get_landsat_8(polygon, start_date):
    ee_geom = ee_utils.gdf_to_ee_polygon(polygon)
    ee_start_date = ee.Date(start_date)
    ee_end_date = ee_start_date.advance(1, 'year')

    ls8SR = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    ls8 = ls8SR.map(mask_cloud_pixels_landsat_8).map(add_NDVI)

    pre_sample_ls = ls8.filterBounds(ee_geom) \
        .filterDate(ee_start_date, ee_end_date) \
        .filter(ee.Filter.dayOfYear(START_DAY, END_DAY)) \
        .mean() \
        .select(['SR_B.', 'NDVI']) \
        .cast({'NDVI': 'double'})

    return pre_sample_ls


def get_landsat_5(polygon, start_date):
    ee_geom = ee_utils.gdf_to_ee_polygon(polygon)
    ee_start_date = ee.Date(start_date)
    ee_end_date = ee_start_date.advance(1, 'year')

    ls5SR = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
    ls8 = ls5SR.map(mask_cloud_pixels_landsat_5).map(add_NDVI_L5)

    pre_sample_ls = ls8.filterBounds(ee_geom) \
        .filterDate(ee_start_date, ee_end_date) \
        .filter(ee.Filter.dayOfYear(START_DAY, END_DAY)) \
        .mean() \
        .select(['SR_B.', 'NDVI']) \
        .cast({'NDVI': 'double'})

    return pre_sample_ls


def get_landsat_7(polygon, start_date):
    ee_geom = ee_utils.gdf_to_ee_polygon(polygon)
    ee_start_date = ee.Date(start_date)
    ee_end_date = ee_start_date.advance(1, 'year')

    ls5SR = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
    ls8 = ls5SR.map(mask_cloud_pixels_landsat_5).map(add_NDVI_L5)

    pre_sample_ls = ls8.filterBounds(ee_geom) \
        .filterDate(ee_start_date, ee_end_date) \
        .filter(ee.Filter.dayOfYear(START_DAY, END_DAY)) \
        .mean() \
        .select(['SR_B.', 'NDVI']) \
        .cast({'NDVI': 'double'})

    return pre_sample_ls

# Another attempt at various LANDSAT metrics.


def get_landsat_advanced(polygon, start_date):
    ee_geom = ee_utils.gdf_to_ee_polygon(polygon)
    ee_start_date = ee.Date(start_date)
    ee_end_date = ee_start_date.advance(1, 'year')

    ls8SR = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    ls8 = ls8SR.map(mask_cloud_pixels)

    landsat_ic = ls8.filterBounds(ee_geom) \
        .filterDate(ee_start_date, ee_end_date)

    landsat_ic = landsat_ic.map(applyScaleFactors).select('SR_B.')
    landsat_ic = landsat_ic.map(add_NDVI).map(add_NDWI).map(add_NBR) \
        .map(add_NDMI).map(add_SWIRS).map(add_SVVI).map(add_TCT)

    bands = ['SR_B1',
             'SR_B2',
             'SR_B3',
             'SR_B4',
             'SR_B5',
             'SR_B6',
             'SR_B7',
             'NDVI',
             'NDWI',
             'NBR',
             'NDMI',
             'SWIRS',
             'SVVI',
             'brightness',
             'greenness',
             'wetness']

    landsat_ic.select(bands)
    return landsat_ic.select(bands).mean()
    reducers = ee.Reducer.mean().combine(
        ee.Reducer.stdDev(),
        sharedInputs=True
    )
    # .combine(
    #    ee.Reducer.median(),
    #    sharedInputs=True
    # )
    return landsat_ic.reduce(reducers)
    # mean_img=landsat_ic.reduce(ee.Reducer.mean())
    # median_img=landsat_ic.reduce(ee.Reducer.median())
    # return mean_img.addBands(median_img)
    # return std_img.addBands(mean_img)
    return std_img


def add_NDVI_L5(image):
    ndvi = image.normalizedDifference(['SR_B4', 'SR_B3']).rename('NDVI')
    return image.addBands(ndvi)


def add_NDVI(image):
    ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI') \
        .cast({'NDVI': 'double'})
    return image.addBands(ndvi)


def add_NDWI(image):
    ndwi = image.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI') \
        .cast({'NDWI': 'double'})
    return image.addBands(ndwi)


def add_NBR(image):
    nbr = image.normalizedDifference(['SR_B5', 'SR_B7']).rename('NBR') \
        .cast({'NBR': 'double'})
    return image.addBands(nbr)


def add_NDMI(image):
    ndmi = image.normalizedDifference(['SR_B5', 'SR_B6']).rename('NDMI') \
        .cast({'NDMI': 'double'})
    return image.addBands(ndmi)


def add_SWIRS(image):
    swirs = image.normalizedDifference(['SR_B6', 'SR_B7']).rename('SWIRS') \
        .cast({'SWIRS': 'double'})
    return image.addBands(swirs)


def add_SVVI(image):
    stdev = image.addBands(
        image.select("SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5",
                     "SR_B7").reduce(ee.Reducer.stdDev()).rename("stdev_B1-B7")) \
        .round().addBands(
        image.select("SR_B4", "SR_B5", "SR_B7").reduce(
            ee.Reducer.stdDev()).rename("stdev_B4-B7")
        .round())
    svvi = stdev.addBands(stdev.select("stdev_B1-B7").subtract(stdev.select("stdev_B4-B7"))
                          .rename("SVVI"))
    return image.addBands(svvi.select("SVVI"))


def add_TCT(image):
    b = image.select("SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7")
    # Coefficients are only for Landsat 8 TOA
    brightness_coefficents = ee.Image(
        [0.3029, 0.2786, 0.4733, 0.5599, 0.508, 0.1872])
    greenness_coefficents = ee.Image(
        [-0.2941, -0.243, -0.5424, 0.7276, 0.0713, -0.1608])
    wetness_coefficents = ee.Image(
        [0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559])

    brightness = image.expression(
        '(B * BRIGHTNESS)', {
            'B': b,
            'BRIGHTNESS': brightness_coefficents
        }
    )
    greenness = image.expression(
        '(B * GREENNESS)',
        {
            'B': b,
            'GREENNESS': greenness_coefficents
        }
    )
    wetness = image.expression(
        '(B * WETNESS)',
        {
            'B': b,
            'WETNESS': wetness_coefficents
        }
    )
    brightness = brightness.reduce(ee.call("Reducer.sum")).rename('brightness')
    greenness = greenness.reduce(ee.call("Reducer.sum")).rename('greenness')
    wetness = wetness.reduce(ee.call("Reducer.sum")).rename('wetness')
    return image.addBands(brightness).addBands(greenness).addBands(wetness)


def applyScaleFactors(image):
    opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    return image.addBands(opticalBands, None, True)
