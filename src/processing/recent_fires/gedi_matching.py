'''
Code for matching spatial matching of GEDI shots that are 'nearby' of each
other. This allows us to roughly sample the same region at different times.
'''

import geopandas as gpd
import pandas as pd
from src.data.fire_perimeters import Fire
from src.data import k_nn
from src.data import pai_vertical
import datetime


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
        result[f'{col}_diff'] = result[col] - result[f'{col}_after']
        result[f'{col}_rel'] = result[f'{col}_after']/result[col]

    return result


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

    # For each shot in before fire, find the closest shot after fire.
    # Break it down per severity.
    result_low = find_matches(before_fire, after_fire, 2, column)
    result_medium = find_matches(before_fire, after_fire, 3, column)
    result_high = find_matches(before_fire, after_fire, 4, column)

    result = pd.concat([result_low, result_medium, result_high])
    result['start_offset'] = start_offset
    result['end_offset'] = end_offset
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

    before_fire = pai_vertical.transform_pai_z(before_fire)
    after_fire = pai_vertical.transform_pai_z(after_fire)

    # For each shot in before fire, find the closest shot after fire.
    # Break it down per severity.
    result_low = find_matches(before_fire, after_fire, 2, 'pai_z_delta_np')
    result_medium = find_matches(before_fire, after_fire, 3, 'pai_z_delta_np')
    result_high = find_matches(before_fire, after_fire, 4, 'pai_z_delta_np')

    result = pd.concat([result_low, result_medium, result_high])
    result['start_offset'] = start_offset
    result['end_offset'] = end_offset
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
    return df[df.burn_severity_median == severity]


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
