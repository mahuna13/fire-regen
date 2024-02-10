import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def divide_gedi_data_into_a_grid(r, shape, df):
    minx, miny, maxx, maxy = shape.geometry.bounds.iloc[0]

    # convert distance to meters
    width_m = (maxx - minx) * 111139
    length_m = (maxy - miny) * 111139

    # calculate how many cells per side
    width_num_steps = width_m // r
    length_num_steps = length_m // r

    # Calculate step size
    stepx = (maxx - minx) / width_num_steps
    stepy = (maxy - miny) / length_num_steps

    df['x'] = ((df['longitude'] - minx) / stepx).apply(np.floor)
    df['y'] = ((maxy - df['latitude']) / stepy).apply(np.floor)

    return df


def spatial_split_train_and_test_data(gedi, geometry):
    gedi_gridded = divide_gedi_data_into_a_grid(4000, geometry, gedi)
    grouped_xy = gedi_gridded.groupby(['x', 'y']).count() \
        .reset_index()[['x', 'y', 'agbd']] \
        .rename(columns={'agbd': 'shot_count'})

    xy_numpy = grouped_xy.to_numpy()
    X_train, X_test, _, _ = train_test_split(xy_numpy,
                                             xy_numpy,
                                             test_size=0.15,
                                             random_state=0)

    X_test_df = pd.DataFrame(X_test)
    X_test_df.columns = ['x', 'y', 'count']
    X_test_df = X_test_df.drop(columns='count')
    gedi_test = X_test_df.merge(right=gedi_gridded,
                                on=['x', 'y'],
                                how='left')

    X_train_df = pd.DataFrame(X_train)
    X_train_df.columns = ['x', 'y', 'count']
    X_train_df = X_train_df.drop(columns='count')

    gedi_train = X_train_df.merge(right=gedi_gridded,
                                  on=['x', 'y'],
                                  how='left')

    return gedi_test, gedi_train
