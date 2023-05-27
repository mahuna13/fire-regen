import ee
import numpy as np
import shapely
from src.utils.logging_util import get_logger
import time

GDRIVE_FOLDER_NAME = 'fire_regen'
logger = get_logger(__file__)


def gdf_to_ee_polygon(gdf_polygon: shapely.Polygon) -> ee.Geometry.Polygon:
    ''' Helper to convert GeoPandas geometry to Earth Engine geometry. '''
    x, y = gdf_polygon.exterior.coords.xy
    coords = np.dstack((x, y)).tolist()
    return ee.Geometry.Polygon(coords)


def save_image_to_drive(image: ee.Image, polygon: shapely.Polygon, img_name: str, scale: int, debug: bool = False):
    ''' Creates a task to save ee.Image to Google Drive as a tif. '''
    ee_geom = gdf_to_ee_polygon(polygon)
    task = ee.batch.Export.image.toDrive(**{
        'image': image,
        'description': img_name,
        'fileNamePrefix': img_name,
        'folder': GDRIVE_FOLDER_NAME,
        'scale': scale,
        'region': ee_geom.getInfo()['coordinates'],
        'maxPixels': 500000000
    })
    task.start()

    if not debug:
        return

    while task.active():
        logger.debug('Polling for task (id: {}).'.format(task.id))
        time.sleep(5)
    logger.debug(task.status)
    logger.debug(ee.data.listOperations())


def save_image_to_drive_per_band(image: ee.Image, polygon: shapely.Polygon, img_name: str, scale: int):
    band_list = image.bandNames().getInfo()
    for i in range(len(band_list)):
        band_name = band_list[i]
        export_img = image.select(band_name)
        save_image_to_drive(export_img, polygon,
                            img_name + '_' + band_name, scale)
