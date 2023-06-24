import ee
import shapely
from src.data.ee import ee_utils


def get_land_cover(start_date: str, end_date: str, polygon: shapely.Polygon) -> ee.Image:
    region = ee_utils.gdf_to_ee_polygon(polygon)

    dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
        .filterDate(start_date, end_date) \
        .filterBounds(region)

    classification = dw.select('label')
    dwComposite = classification.reduce(ee.Reducer.mode())

    return dwComposite
