import ee
from src.data import ee_utils


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


def mask_cloud_pixels(ls_img: ee.Image):
    qaMask = ls_img.select(['QA_PIXEL']).bitwiseAnd(int('11111', 2)).eq(0)

    return ls_img.updateMask(qaMask) \
        .copyProperties(ls_img, ['system:time_start'])


def get_landsat(polygon, sample_date):
    ee_geom = ee_utils.gdf_to_ee_polygon(polygon)
    ee_sample_date = ee.Date(sample_date)
    ee_pre_sample_date = ee_sample_date.advance(-1, 'year')

    ls8SR = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    ls8 = ls8SR.map(mask_cloud_pixels)

    pre_sample_ls = ls8.filterBounds(ee_geom) \
        .filterDate(ee_pre_sample_date, ee_sample_date) \
        .filter(ee.Filter.dayOfYear(START_DAY, END_DAY)) \
        .mean() \
        .select('SR_B.')

    pre_sample_ls = pre_sample_ls.addBands(get_NBR_index(pre_sample_ls))
    pre_sample_ls = pre_sample_ls.addBands(get_NDVI_index(pre_sample_ls))

    return pre_sample_ls
