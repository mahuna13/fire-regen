import ee


def get_burn_severity_data():
    yearly_bs = ee.ImageCollection(
        'USFS/GTAC/MTBS/annual_burn_severity_mosaics/v1')

    # Filter data to get only continental US.
    conus = yearly_bs.filter(
        ee.Filter.stringContains('system:index', 'mtbs_CONUS'))
    return _add_year_as_band(conus)


def get_burn_count_data():
    severity = get_burn_severity_data()

    zero_burn = ee.Image.constant(0).rename('burn_count')

    def mask_for_fire_existence(image):
        gt_1 = image.select('Severity').gt(1)
        lt_5 = image.select('Severity').lt(5)
        return gt_1.And(lt_5).rename('burn_count')

    burn_count_from_severity = severity.map(mask_for_fire_existence).sum()
    return zero_burn.add(burn_count_from_severity.unmask())


def _add_year_as_band(ic: ee.ImageCollection):
    def add_year(image):
        year = ee.Number.parse(
            ee.String(image.get('system:index')).split('_').get(-1))
        year_band = ee.Image.constant(year).rename('year').toInt() \
            .updateMask(image.mask())
        return image.addBands(year_band)

    return ic.map(add_year)
