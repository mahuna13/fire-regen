import pandas as pd
import numpy as np


def transform_pai_z(df: pd.DataFrame):
    '''
    Transforms the original pai_z array into a more useful set of columns:
    pai_z_np - numpy array representing cumulative vertical pai starting from
    the lowest heights, pai_z_padded - same as pai_z_np but padded with zeros
    so that all the arrays are of the same length of 30, pai_z_delta_np -
    delta PAI for each height bucket.

    NOTE: This function can run for a long time (~12 mins) for ~10 million
    footprints.
    '''
    df = transform_pai_z_array_to_np(df)
    df = pad_all_pai_z_to_constant_length(df)
    df = calculate_pai_z_delta(df)
    return df


def transform_pai_z_array_to_np(df: pd.DataFrame):
    '''
    For some reason, pai_z is a string, so convert to a numpy array,
    flip it around (so that the lowest heights come first in the array),
    and trim zeros.
    '''
    df['pai_z_np'] = df.apply(lambda row: np.flip(np.trim_zeros(
        np.array(row.pai_z))), axis=1)
    return df


def pad_all_pai_z_to_constant_length(df: pd.DataFrame):
    def pad_numpy_array(df):
        pai_array = df.pai_z_np
        return np.pad(pai_array, (0, 30-pai_array.size), 'constant')

    df['pai_z_padded'] = df.apply(pad_numpy_array, axis=1)
    return df


def pad_array(df: pd.DataFrame, column: str, constant_values=(0, 0)):
    def pad_numpy_array(df):
        pai_array = df[column]
        return np.pad(pai_array, (0, 30-pai_array.size), 'constant', constant_values=constant_values)

    df[f'{column}_padded'] = df.apply(pad_numpy_array, axis=1)
    return df


def calculate_pai_z_delta(df: pd.DataFrame):
    def calc_delta(df):
        pai_array = df.pai_z_padded
        pai_rolled = np.roll(pai_array, 1)
        pai_delta = pai_array - pai_rolled
        pai_delta[pai_delta < 0] = 0
        return pai_delta

    df['pai_z_delta_np'] = df.apply(calc_delta, axis=1)
    return df


def calculate_pai_z_percentage(df: pd.DataFrame):
    def calc_percentage(df):
        pai_array = df.pai_z_np
        pai = df.pai
        return np.around((pai_array / pai) * 100, 1)

    df['pai_z_percent'] = df.apply(calc_percentage, axis=1)
    return df


def maximum_pai_height(df: pd.DataFrame):
    df['pai_max_height'] = df.apply(lambda row: len(row.pai_z) * 5, axis=1)
    return df

# TODO: Consider doing this if needed.


def first_element_larger_than(l: list[float], t: float):
    for i in range(len(l)):
        if l[i] <= t:
            return i
    return


def transform_pai_z_2(df: pd.DataFrame):
    df = transform_pai_z_array_to_np(df)
    df = calculate_pai_z_percentage(df)
    df["pai_z"] = df.pai_z_np
    df = df.drop(columns=["pai_z_np"])
    df = maximum_pai_height(df)
    df = pad_array(df, "pai_z")
    df = pad_array(df, "pai_z_percent")
    return df
