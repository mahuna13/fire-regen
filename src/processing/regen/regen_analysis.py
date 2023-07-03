import geopandas as gpd
import pandas as pd
from src.data.fire_perimeters import Fire, MTBSFire
from src.data import gedi_raster_matching
from src.data.ee import lcms_import
import numpy as np
from fastai.tabular.all import load_pickle
from src.constants import DATA_PATH
import pickle
import shapely


def process_all_fires_with_rf_control_per_burn_year(inference_path, model_path=f"{DATA_PATH}/rf/models") -> gpd.GeoDataFrame:
    # Process year by year
    fire_shots = []
    for year in range(1985, 2021):
        print(f'Process fires for year {year}.')

        # Load model and data for that year.
        m = load_pickle(f"{model_path}/model_{year}.pkl")
        to = pickle.load(
            open(f"{model_path}/to_{year}.pkl", 'rb'))
        gedi = load_pickle(f"{inference_path}/gedi_match_{year}.pkl")

        print("matching shots within fire")
        processed = gedi[gedi.burn_year == year]

        # Get AGBD control from RF.
        print("Processing data")
        to_new = to.train.new(processed)
        to_new.process()

        print('Run RF to predict control AGBD.')
        processed['agbd_control'] = np.exp(m.predict(to_new.train.xs))
        processed['rel_agbd'] = processed.agbd / processed.agbd_control

        result = processed[['shot_number', 'longitude', 'latitude', 'agbd', 'agbd_pi_lower',
                            'agbd_pi_upper', 'agbd_se', 'beam_type', 'sensitivity', 'pft_class',
                            'gedi_year', 'gedi_month', 'absolute_time', 'geometry',
                            'severity', 'burn_year', 'burn_count',
                            'time_since_burn',  'elevation', 'slope', 'aspect',
                            'soil', 'agbd_control', 'rel_agbd']]

        fire_shots.append(result)
    return pd.concat(fire_shots)


def process_all_fires_with_rf_control(
    perimeters: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    # Process year by year
    fire_shots = []
    for year in range(1989, 2021):
        print(f'Process fires for year {year}.')
        fires = perimeters[perimeters.YEAR_ == str(year)]

        # Load model and data for that year.
        m = load_pickle(f"{DATA_PATH}/rf/models/model_{year}.pkl")
        to = pickle.load(open(f"{DATA_PATH}/rf/models/to_{year}.pkl", 'rb'))
        gedi = load_pickle(
            f"{DATA_PATH}/rf/burned_gedi/gedi_match_{year}.pkl")

        for perimeter in fires.itertuples():
            print(
                f'Processing fire {perimeter.FIRE_NAME} and {perimeter.Index}')
            fire = Fire(perimeters[perimeters.index == perimeter.Index])

            print("matching shots within fire")
            within_perimeter = gedi.sjoin(
                fire.fire, how="inner", predicate="within")

            if within_perimeter.empty:
                print(f"No matches for fire {perimeter.FIRE_NAME}.")
                continue

            # Step 1. Calculate distance to fire perimeter.
            print("Calculate distance to perimeter.")
            processed = distance_to_perimeter(
                fire.fire.geometry, within_perimeter)

            # Get AGBD control from RF.
            print("Processing data")
            to_new = to.train.new(processed)
            to_new.process()

            print('Run RF to predict control AGBD.')
            processed['agbd_control'] = m.predict(to_new.train.xs)

            result = processed[['shot_number', 'longitude', 'latitude', 'agbd', 'agbd_pi_lower',
                                'agbd_pi_upper', 'agbd_se', 'beam_type', 'sensitivity', 'pft_class',
                                'gedi_year', 'gedi_month', 'absolute_time', 'geometry',
                                'burn_severity_median', 'burn_year_median', 'burn_counts_median',
                                'time_since_burn',  'elevation', 'slope', 'aspect',
                                'soil', 'distance_to_perimeter', 'agbd_control', 'YEAR_', 'FIRE_NAME', 'INC_NUM', 'Shape_Area']]

            fire_shots.append(result)
    return pd.concat(fire_shots)


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


def match_with_unburned_control_using_rf(
    fire: MTBSFire,
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

    within_fire = gedi_burned.sjoin(fire.fire, how="inner", predicate="within")

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


def match_with_unburned_control_using_rf(fire):
    model_path = f"{DATA_PATH}/rf/models_shallow"
    inference_path = f"{DATA_PATH}/rf/burned/3x3"
    unburned_path = f"{DATA_PATH}/rf/unburned_B"

    year = fire.fire.Ig_Date.dt.year.iloc[0]
    if year <= 1984 or year > 2020:
        print(f"{year} out of range")
        return

    m = load_pickle(f"{model_path}/model_{year}.pkl")
    to = pickle.load(open(f"{model_path}/to_{year}.pkl", 'rb'))
    gedi_burned = load_pickle(f"{inference_path}/gedi_match_{year}.pkl")
    gedi_unburned = load_pickle(f"{unburned_path}/gedi_match_{year}.pkl")

    buffer = fire.get_buffer(5000, 100)
    within_buffer = gedi_unburned.sjoin(
        buffer, how="inner", predicate="within")
    within_fire = gedi_burned[gedi_burned.burn_year == year].sjoin(
        fire.fire, how="inner", predicate="within")

    if within_fire.empty or within_buffer.empty:
        print(f"No matching burn shots")
        return

    to_burned = to.train.new(within_fire)
    to_burned.process()

    to_unburned = to.train.new(within_buffer)
    to_unburned.process()

    terminals_burned = m.apply(to_burned.train.xs)
    terminals_unburned = m.apply(to_unburned.train.xs)

    a = terminals_burned[:, 0]
    b = terminals_unburned[:, 0]

    proxMat = 1*np.equal.outer(a, b)

    for i in range(1, 100):
        a = terminals_burned[:, i]
        b = terminals_unburned[:, i]
        proxMat += 1*np.equal.outer(a, b)

    proxMat = proxMat / 100

    proxMat_max = np.argsort(proxMat, axis=1)[:, -10:]

    control_agbd = np.empty(proxMat_max.shape[0])
    for i in range(0, proxMat_max.shape[0]):
        control_agbd[i] = within_buffer.iloc[proxMat_max[i, :]].agbd.mean()

    within_fire["control_agbd"] = control_agbd
    within_fire["rel_agbd"] = within_fire.agbd / within_fire.control_agbd

    return within_fire


def match_with_unburned_control_using_rf_2(fire, m, to, gedi_burned, gedi_unburned):

    buffer = fire.get_buffer(5000, 100)
    within_buffer = gedi_unburned.sjoin(
        buffer, how="inner", predicate="within")
    within_fire = gedi_burned.sjoin(
        fire.fire, how="inner", predicate="within")

    if within_fire.empty or within_buffer.empty:
        print(f"No matching burn shots")
        return

    to_burned = to.train.new(within_fire)
    to_burned.process()

    to_unburned = to.train.new(within_buffer)
    to_unburned.process()

    terminals_burned = m.apply(to_burned.train.xs)
    terminals_unburned = m.apply(to_unburned.train.xs)

    a = terminals_burned[:, 0]
    b = terminals_unburned[:, 0]

    proxMat = 1*np.equal.outer(a, b)

    for i in range(1, 100):
        a = terminals_burned[:, i]
        b = terminals_unburned[:, i]
        proxMat += 1*np.equal.outer(a, b)

    proxMat = proxMat / 100

    proxMat_max = np.argsort(proxMat, axis=1)[:, -10:]

    control_agbd = np.empty(proxMat_max.shape[0])
    for i in range(0, proxMat_max.shape[0]):
        control_agbd[i] = within_buffer.iloc[proxMat_max[i, :]].agbd.mean()

    within_fire["control_agbd"] = control_agbd
    within_fire["rel_agbd"] = within_fire.agbd / within_fire.control_agbd

    return within_fire


def match_with_proximity_matrix(sierra_perims):
    all_fires = []
    for year in range(1985, 2021):
        print(f"Processing fires in year {year}")
        fires = sierra_perims[sierra_perims.fire_year == year]

        model_path = f"{DATA_PATH}/rf/models_shallow"
        inference_path = f"{DATA_PATH}/rf/burned/3x3"
        unburned_path = f"{DATA_PATH}/rf/unburned_B"

        m = load_pickle(f"{model_path}/model_{year}.pkl")
        to = pickle.load(open(f"{model_path}/to_{year}.pkl", 'rb'))
        gedi_burned = load_pickle(f"{inference_path}/gedi_match_{year}.pkl")
        gedi_burned = gedi_burned[gedi_burned.burn_year == year]
        gedi_unburned = load_pickle(f"{unburned_path}/gedi_match_{year}.pkl")
        for perimeter in fires.itertuples():
            fire = MTBSFire(fires[fires.index == perimeter.Index])
            print(f"Processing fire {fire.fire.Incid_Name}")
            all_fires.append(match_with_unburned_control_using_rf_2(
                fire, m, to, gedi_burned, gedi_unburned))

    return pd.concat(all_fires)


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

    if isinstance(perimeter_3310.geometry.iloc[0],
                  shapely.geometry.multipolygon.MultiPolygon):
        polygons = list(perimeter_3310.geometry.iloc[0].geoms)
    else:
        polygons = [perimeter_3310.geometry.iloc[0]]
    # polygons = list(perimeter_3310.geometry.iloc[0].geoms)
    distances = np.empty((gedi.shape[0], len(polygons)))
    for idx in range(len(polygons)):
        distances[:, idx] = gedi_3310.distance(polygons[idx].exterior)

    gedi['distance_to_perimeter'] = distances.min(axis=1)
    return gedi
