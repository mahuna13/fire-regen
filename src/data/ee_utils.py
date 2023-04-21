import ee
import numpy as np
import shapely
import time

GDRIVE_FOLDER_NAME = 'fire_regen'


def gdf_to_ee_polygon(gdf_polygon: shapely.Polygon) -> ee.Geometry.Polygon:
    ''' Helper to convert GeoPandas geometry to Earth Engine geometry. '''
    x, y = gdf_polygon.exterior.coords.xy
    coords = np.dstack((x, y)).tolist()
    return ee.Geometry.Polygon(coords)


def save_image_to_drive(image: ee.Image, polygon: shapely.Polygon, img_name: str, scale: int):
    ''' Creates a task to save ee.Image to Google Drive as a tif. '''
    ee_geom = gdf_to_ee_polygon(polygon)
    task = ee.batch.Export.image.toDrive(**{
        'image': image,
        'description': img_name,
        'folder': GDRIVE_FOLDER_NAME,
        'scale': 30,
        'region': ee_geom.getInfo()['coordinates']
    })
    task.start()

    while task.active():
        print('Polling for task (id: {}).'.format(task.id))
        time.sleep(5)
    print(task.status)
    print(ee.data.listOperations())
