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
        np.array(row.pai_z[1:-1].split(", ")).astype(float))), axis=1)
    return df


def pad_all_pai_z_to_constant_length(df: pd.DataFrame):
    def pad_numpy_array(df):
        pai_array = df.pai_z_np
        return np.pad(pai_array, (0, 30-pai_array.size), 'constant')

    df['pai_z_padded'] = df.apply(pad_numpy_array, axis=1)
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
