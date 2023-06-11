import geopandas as gpd
import pandas as pd
from src.data.fire_perimeters import Fire
from src.data import gedi_raster_matching
from src.data.ee import lcms_import
import numpy as np


def process_all_fires(
    perimeters: gpd.GeoDataFrame,
    gedi_burned: gpd.GeoDataFrame,
    gedi_unburned: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    '''
    Process all GEDI shots that burned in the last 30 years.

    GEDI shots should already be matched with MTBS burned mosaic, and
    preprocessed for regrowth analysis.

    For each fire:
       * Intersect GEDI shots with each individual fire perimeter.
       * Calculate distance to fire perimeter.
       * Match with land cover.
       * Match with climate data.
       * Find unburned control for each burned pixel.
    '''
    fire_shots = []
    for perimeter in perimeters.itertuples():
        print(f'Processing fire {perimeter.FIRE_NAME} and {perimeter.Index}')
        fire = Fire(perimeters[perimeters.index == perimeter.Index])

        within_perimeter = gedi_burned.sjoin(
            fire.fire, how="inner", predicate="within")

        if within_perimeter.empty:
            print(f"No matches for fire {perimeter.FIRE_NAME}.")
            continue

        # Step 1. Calculate distance to fire perimeter.
        processed = distance_to_perimeter(fire.fire.geometry, within_perimeter)

        # Step 3. Get climate data since burn.
        # processed = match_climate_data(processed)

        # Step 4. Find a control, unburned GEDI shot for each burned shot.
        processed = match_with_unburned_control(
            fire, processed, gedi_unburned)

        fire_shots.append(processed)

    result = pd.concat(fire_shots)

    return result


def match_with_unburned_control(
    fire: Fire,
    gedi_burned: gpd.GeoDataFrame,
    gedi_unburned: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    # TODO: implement.
    # Get the unburned buffer around the fire, 5km around the fire, excluding
    # 100m around the fire.
    buffer = fire.get_buffer(5000, 100)

    # Find unburned shots within the buffer.
    within_buffer = gedi_unburned.sjoin(
        buffer, how="inner", predicate="within")

    # Find shots that were trees before the fire.
    # TODO: pick the ones that were and still are trees?
    burn_year = fire.alarm_date.year

    within_buffer_lc = gedi_raster_matching.match_landcover_for_year(
        burn_year, within_buffer, 3)
    within_buffer_trees = within_buffer_lc[
        (within_buffer_lc.land_cover_std == 0) &
        (within_buffer_lc.land_cover_median == 1)]

    # Pick 200 random shots and assign as control. Do both median and mean.
    random_controls = within_buffer_trees.sample(
        min(200, within_buffer_trees.shape[0]))
    gedi_burned['agbd_control_mean'] = random_controls.agbd.mean()
    gedi_burned['agbd_control_median'] = random_controls.agbd.median()

    return gedi_burned


def match_climate_data(
    gedi: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    # TODO: implement.
    return gedi


def distance_to_perimeter(
    perimeter,
    gedi: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    '''
    For each GEDI shot in gedi, calculate the shortest distance to the
    perimeter.

    Returns the same gedi df with the added column - 'distance_to_perimeter'.
    '''
    gedi_3310 = gedi.to_crs(epsg=3310)
    perimeter_3310 = perimeter.to_crs(epsg=3310)

    polygons = list(perimeter_3310.geometry.iloc[0].geoms)
    distances = np.empty((gedi.shape[0], len(polygons)))
    for idx in range(len(polygons)):
        distances[:, idx] = gedi_3310.distance(polygons[idx].exterior)

    gedi['distance_to_perimeter'] = distances.min(axis=1)
    return gedi
