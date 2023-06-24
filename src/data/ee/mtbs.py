import ee


def get_aggregated_burn_data() -> ee.Image:
    '''
    Fetches MTBS burn severity data from Earth Engine, and augments it with
    additional derived data.

    Returns a single Image that has three bands,
    'last_burn_severity', 'last_burn_year' and 'burn_count'.
    '''
    severity = get_burn_severity_data().sort('system:time_start', True)
    burn_count = get_burn_count_data(severity)

    # Doing mosaic will return an Image where each pixel contains the severity
    # and year at which it burned last i.e. information about the latest burn
    # within that pixel.
    latest_burn = severity.mosaic().rename('last_burn_severity',
                                           'last_burn_year')

    # Combine severity, year and count.
    aggregated_burn_img = latest_burn.addBands(burn_count)

    # Cast bands to consistent types, so that the tif image can be saved
    # to Google Drive.
    return aggregated_burn_img.cast({'last_burn_year': 'uint16',
                                     'last_burn_severity': 'uint16',
                                     'burn_count': 'uint16'})


def get_burn_severity_data() -> ee.ImageCollection:
    '''
    Fetches MTBS burn severity data from Earth Engine.

    Returns ImageCollection with two bands, 'burn_severity' and 'burn_year',
    for continental US.

    Each image in image collection represents severity at which each pixel 
    burned that year. 'burn_year' was placed as an additional band to
    fascilitate down the stream querying and processing.
    '''
    yearly_bs = ee.ImageCollection(
        'USFS/GTAC/MTBS/annual_burn_severity_mosaics/v1') \
        .select(['Severity'], ['burn_severity'])  # rename to burn_severity

    # Filter data to get only continental US.
    conus = yearly_bs.filter(
        ee.Filter.stringContains('system:index', 'mtbs_CONUS'))
    return _add_year_as_band(conus)


def get_burn_severity_data_for_year(year: int) -> ee.Image:
    return get_burn_severity_data().filterDate(str(year), str(year + 1)).first()


def get_burn_count_data(ic_severity: ee.ImageCollection) -> ee.Image:
    zero_burn = ee.Image.constant(0).rename('burn_count')

    def mask_for_fire_existence(image):
        gt_1 = image.select('burn_severity').gt(1)
        lt_5 = image.select('burn_severity').lt(5)
        return gt_1.And(lt_5).rename('burn_count')

    burn_count_from_severity = ic_severity.map(mask_for_fire_existence).sum()
    return zero_burn.add(burn_count_from_severity.unmask())


def _add_year_as_band(ic: ee.ImageCollection):
    def add_year(image):
        year = ee.Number.parse(
            ee.String(image.get('system:index')).split('_').get(-1))
        year_band = ee.Image.constant(year).rename('burn_year').toInt() \
            .updateMask(image.mask())
        return image.addBands(year_band)

    return ic.map(add_year)
