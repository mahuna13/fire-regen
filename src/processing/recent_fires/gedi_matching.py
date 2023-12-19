'''
Code for matching spatial matching of GEDI shots that are 'nearby' of each
other. This allows us to roughly sample the same region at different times.
'''

import geopandas as gpd
import numpy as np
import pandas as pd
from src.data.fire_perimeters import Fire, FirePerimeters
from src.data import k_nn
from src.data import pai_vertical
import datetime
import importlib
importlib.reload(pai_vertical)


def find_matches(
        left: gpd.GeoDataFrame,
        right: gpd.GeoDataFrame,
        severity: int = None,
        column: str = None,
        columns: list[str] = None
) -> gpd.GeoDataFrame:
    '''
    For each row in the left GeoDataFrame, it finds the closest row in the 
    right GeoDataFrame, and extracts column values from the matching row.
    '''
    if severity is None:
        left_input = left
        right_input = right
    else:
        left_input = get_severity(left, severity)
        right_input = get_severity(right, severity)

    if left_input.shape[0] == 0 or right_input.shape[0] == 0:
        # No shots available for matching.
        return None

    closest_indeces, distances = k_nn.nearest_neighbors(
        left_input, right_input, 1)

    result = left_input.copy()
    result['closest_distance'] = distances
    result[f'match_datetime'] = pd.to_datetime(
        right_input.iloc[closest_indeces.flatten()
                         ].absolute_time.values, utc=True, format='mixed')

    if column is not None:
        columns = [column]

    for col in columns:
        result[f'{col}_after'] = right_input.iloc[closest_indeces.flatten()
                                                  ][col].values
        result[f'{col}_diff'] = result[f'{col}_after'] - result[col]
        result[f'{col}_rel'] = result[f'{col}_after']/result[col]

    return result


def match_measurements_before_and_after_fire_no_severity(
        fire: Fire,
        gedi: gpd.GeoDataFrame,
        column: str,
        start_offset: int = 0,
        end_offset: int = None
) -> gpd.GeoDataFrame:
    within_fire_perimeter = gedi.sjoin(
        fire.fire, how="inner", predicate="within")

    before_fire = get_shots_before_date(fire.alarm_date, within_fire_perimeter)
    after_fire = get_shots_after_date(
        fire.cont_date, within_fire_perimeter, start_offset, end_offset)

    if before_fire.shape[0] == 0 or after_fire.shape[0] == 0:
        # No GEDI shots at the fire area found.
        return None

    # For each shot in before fire, find the closest shot after fire.
    # Break it down per severity.
    return find_matches(before_fire, after_fire, column=column)


def match_measurements_before_and_after_fire(
        fire: Fire,
        gedi: gpd.GeoDataFrame,
        column: str,
        start_offset: int = 0,
        end_offset: int = None
) -> gpd.GeoDataFrame:
    within_fire_perimeter = gedi.sjoin(
        fire.fire, how="inner", predicate="within")

    before_fire = get_shots_before_date(fire.alarm_date, within_fire_perimeter)
    after_fire = get_shots_after_date(
        fire.cont_date, within_fire_perimeter, start_offset, end_offset)

    if before_fire.shape[0] == 0 or after_fire.shape[0] == 0:
        # No GEDI shots at the fire area found.
        return None

    # For each shot in before fire, find the closest shot after fire.
    # Break it down per severity.
    result_low = find_matches(before_fire, after_fire, 2, column)
    result_medium = find_matches(before_fire, after_fire, 3, column)
    result_high = find_matches(before_fire, after_fire, 4, column)

    if result_low is None and result_medium is None and result_high is None:
        return None

    result = pd.concat([result_low, result_medium, result_high])
    result['start_offset'] = start_offset
    result['end_offset'] = end_offset
    result['cont_date'] = fire.cont_date
    result['date_since'] = (
        (result.match_datetime - fire.cont_date)/np.timedelta64(1, 'M')).astype(int)
    return result


def match_measurements_before_and_after_date(
        date: str,
        gedi: gpd.GeoDataFrame,
        column: str,
        start_offset: int = 0,
        end_offset: int = None
) -> gpd.GeoDataFrame:
    pd_date = pd.to_datetime(date, utc=True)
    before_date = get_shots_before_date(pd_date, gedi)
    after_date = get_shots_after_date(
        pd_date, gedi, start_offset, end_offset)

    # For each shot in before fire, find the closest shot after fire.
    # Break it down per severity.
    result = find_matches(before_date, after_date, column=column)
    result['start_offset'] = start_offset
    result['end_offset'] = end_offset
    return result


def match_pai_z_before_and_after_date(
        date: str,
        gedi: gpd.GeoDataFrame,
        start_offset: int = 0,
        end_offset: int = None,
) -> gpd.GeoDataFrame:
    pd_date = pd.to_datetime(date, utc=True)
    before_date = get_shots_before_date(pd_date, gedi)
    after_date = get_shots_after_date(
        pd_date, gedi, start_offset, end_offset)

    before_date = pai_vertical.transform_pai_z(before_date)
    after_date = pai_vertical.transform_pai_z(after_date)

    # For each shot in before fire, find the closest shot after fire.
    # Break it down per severity.
    result = find_matches(before_date, after_date, column='pai_z_delta_np')
    result['start_offset'] = start_offset
    result['end_offset'] = end_offset
    return result


def match_across_fire_perimeters(
    firep: gpd.GeoDataFrame,
    gedi: gpd.GeoDataFrame,
    column: str,
    start_offset: int = 0,
    end_offset: int = None
) -> gpd.GeoDataFrame:
    results = []
    for perimeter in firep.itertuples():
        fire = Fire(firep[(firep.INC_NUM == perimeter.INC_NUM) &
                          (firep.FIRE_NAME == perimeter.FIRE_NAME)])
        matches = match_measurements_before_and_after_fire(
            fire, gedi, column, start_offset, end_offset)
        if matches is None:
            print(
                f'Skipped fire {perimeter.FIRE_NAME}. No matching GEDI shots found.')
            continue
        results.append(matches)

    return pd.concat(results)


def match_pai_z_across_fire_perimeters(
    firep: gpd.GeoDataFrame,
    gedi: gpd.GeoDataFrame,
    start_offset: int = 0,
    end_offset: int = None
) -> gpd.GeoDataFrame:
    results = []
    for perimeter in firep.itertuples():
        fire = Fire(firep[(firep.INC_NUM == perimeter.INC_NUM) &
                          (firep.FIRE_NAME == perimeter.FIRE_NAME)])
        matches = match_pai_z_before_and_after_fire(
            fire, gedi, start_offset, end_offset)
        if matches is None:
            print(
                f'Skipped fire {perimeter.FIRE_NAME}. No matching GEDI shots found.')
            continue
        results.append(matches)

    return pd.concat(results)


def match_pai_z_before_and_after_fire(
        fire: Fire,
        gedi: gpd.GeoDataFrame,
        start_offset: int = 0,
        end_offset: int = None,
) -> gpd.GeoDataFrame:
    within_fire_perimeter = gedi.sjoin(
        fire.fire, how="inner", predicate="within")

    before_fire = get_shots_before_date(fire.alarm_date, within_fire_perimeter)
    after_fire = get_shots_after_date(
        fire.cont_date, within_fire_perimeter, start_offset, end_offset)

    if before_fire.empty or after_fire.empty:
        return None

    before_fire = pai_vertical.transform_pai_z(before_fire)
    after_fire = pai_vertical.transform_pai_z(after_fire)

    # For each shot in before fire, find the closest shot after fire.
    # Break it down per severity.
    result_low = find_matches(before_fire, after_fire, 2, 'pai_z_delta_np')
    result_medium = find_matches(before_fire, after_fire, 3, 'pai_z_delta_np')
    result_high = find_matches(before_fire, after_fire, 4, 'pai_z_delta_np')

    if result_low is None and result_medium is None and result_high is None:
        return None

    result = pd.concat([result_low, result_medium, result_high])
    result['start_offset'] = start_offset
    result['end_offset'] = end_offset
    result['cont_date'] = fire.cont_date
    result['date_since'] = (
        (result.match_datetime - fire.cont_date)/np.timedelta64(1, 'M')).astype(int)
    return result


def get_shots_before_date(
    date: datetime.datetime,
    df: pd.DataFrame
):
    return df[df.absolute_time < date]


def get_shots_after_date(
    date: datetime.datetime,
    df: pd.DataFrame,
    start_offset: int,
    end_offset: int
):
    filtered_df = df[df.absolute_time > date +
                     pd.DateOffset(month=start_offset)]

    if end_offset is None:
        return filtered_df

    return filtered_df[df.absolute_time < date +
                       pd.DateOffset(month=end_offset)]


def get_closest_matches(df, distance):
    return df[df.closest_distance < distance]


def get_severity(df, severity):
    return df[df.severity == severity]


def add_date_since_burn_categories_granular(df: gpd.GeoDataFrame):
    c = pd.cut(
        df[['date_since']].stack(),
        [0, 4, 7, 10, 13, 19, 25],
        labels=['burn_3', 'burn_6', "burn_9", "burn_12", "burn_18", "burn_24"]
    )
    return df.join(c.unstack().add_suffix('_cat'))


def add_date_since_burn_categories_coarse(df: gpd.GeoDataFrame):
    c = pd.cut(
        df[['date_since']].stack(),
        [0, 7, 13, 25],
        labels=['burn_6', 'burn_12', "burn_24"]
    )
    return df.join(c.unstack().add_suffix('_cat'))


def add_agbd_rel_ba4_categories(df: gpd.GeoDataFrame):
    c = pd.cut(
        df[['agbd_rel']].stack(),
        [0, 0.25, 0.75, 1, np.inf],
        labels=['4', '3', "2", "1"]
    )
    return df.join(c.unstack().add_suffix('_ba4'))


def add_agbd_rel_ba7_categories(df: gpd.GeoDataFrame):
    c = pd.cut(
        df[['agbd_rel']].stack(),
        [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1, np.inf],
        labels=['7', '6', "5", "4", "3", "2", "1"]
    )
    return df.join(c.unstack().add_suffix('_ba7'))


def add_cover_rel_cc5_categories(df: gpd.GeoDataFrame):
    c = pd.cut(
        df[['cover_rel']].stack(),
        [0, 0.25, 0.5, 0.75, 1, np.inf],
        labels=['5', '4', '3', "2", "1"]
    )
    return df.join(c.unstack().add_suffix('_cc5'))
