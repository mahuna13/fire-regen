import ee
import shapely
import src.data.ee_utils as ee_utils


def get_lcms(
    start_year: str,
    end_year: str,
    polygon: shapely.Polygon
) -> ee.ImageCollection:
    region = ee_utils.gdf_to_ee_polygon(polygon)

    ic = ee.ImageCollection('USFS/GTAC/LCMS/v2021-7')

    lcms = ic.filterDate(start_year, end_year) \
        .filter('study_area == "CONUS"') \
        .filterBounds(region) \
        .select(['Land_Cover'], ['land_cover'])

    return lcms


def get_lcms_for_single_year(
    year: int,
    polygon: shapely.Polygon
) -> ee.Image:
    region = ee_utils.gdf_to_ee_polygon(polygon)
    img = get_lcms(str(year), str(year + 1), polygon).first()
    return img.clip(region)
