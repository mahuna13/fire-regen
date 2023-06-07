from fitter import Fitter, get_common_distributions, get_distributions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

from src.processing.recent_fires.gedi_matching import get_closest_matches


def plot_severity_for_distance(df: pd.DataFrame, col: str, lim: int = 700):
    palette = [sns.color_palette("rocket")[i] for i in [5, 3, 0]]
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    hue = 'burn_severity_median'
    df = df.sort_values(hue, ascending=True)
    df = df[df.burn_severity_median.isin([2, 3, 4])]

    sns.scatterplot(get_closest_matches(df, 5), x=col, y=f'{col}_after',
                    hue=hue, ax=ax[0][0], palette=palette)
    sns.scatterplot(get_closest_matches(df, 10), x=col, y=f'{col}_after',
                    hue=hue, ax=ax[0][1], palette=palette)
    sns.scatterplot(get_closest_matches(df, 20), x=col, y=f'{col}_after',
                    hue=hue, ax=ax[1][0], palette=palette)
    sns.scatterplot(get_closest_matches(df, 40), x=col, y=f'{col}_after',
                    hue=hue, ax=ax[1][1], palette=palette)

    for i in [0, 1]:
        for j in [0, 1]:
            ax[i][j].set_xlim((0, lim))
            ax[i][j].set_ylim((0, lim))


def calculate_error_for_distances(df, distance_range, column_x, column_y):
    all_r2 = []
    all_num_matches = []
    for distance in distance_range:
        matches = get_closest_matches(df, distance)
        if (matches.shape[0] == 0):
            all_r2.append(None)
            all_num_matches.append(0)
        else:
            all_r2.append(r2_score(matches[column_x], matches[column_y]))
            all_num_matches.append(matches.shape[0])

    return all_r2, all_num_matches


def plot_error_for_distances(df, column):
    distances = range(1, 50)

    fig, ax = plt.subplots(3, 2, figsize=(10, 15))
    row_idx = 0
    for severity in [2, 3, 4]:
        r2s, num_matches = calculate_error_for_distances(
            get_severity(df, severity), distances, column, f'{column}_after')
        sns.lineplot(x=distances, y=r2s, ax=ax[row_idx][0])
        sns.lineplot(x=distances, y=num_matches, ax=ax[row_idx][1])
        row_idx += 1


def fit_linear_regression(df, column, ax=None):
    before_values = df[column].values
    after_values = df[f'{column}_after'].values

    fit_best = LinearRegression().fit(np.reshape(
        before_values, (df.shape[0], 1)), after_values)
    best_score = fit_best.score(np.reshape(before_values,
                                           (df.shape[0], 1)), after_values)

    print(
        f'Best Linear regression coefficient is {fit_best.coef_}. \
        R squared is : {best_score}.')

    print(
        f'The error for coeff = 1 is: {r2_score(before_values, after_values)}.'
    )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(
            10, 10), sharex=True, sharey=True)

    sns.scatterplot(df, x=column, y=f'{column}_after', ax=ax)
    sns.lineplot(x=before_values, y=fit_best.predict(np.reshape(
        before_values, (df.shape[0], 1))), color='green', ax=ax)
    sns.lineplot(x=before_values, y=before_values, color='red', ax=ax)


def fit_linear_regression_per_severity(df, column):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    row_idx = 0
    for severity in [2, 3, 4]:
        print(f'Linear regression fit for severity {severity}.')
        fit_linear_regression(get_severity(df, severity), column, ax[row_idx])
        row_idx += 1


def plot_rel_difference_per_severity(df, column):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

    row_idx = 0
    for severity in [2, 3, 4]:
        sns.histplot(get_severity(df, severity),
                     x=f'{column}_rel', ax=ax[row_idx])
        ax[row_idx].set_xlim((0, 2))
        row_idx += 1


def get_severity(df, severity):
    return df[df.burn_severity_median == severity]


def fit_distributions(df, column, rel=False):
    if rel:
        values = df[f'{column}_rel'].values
    else:
        values = df[f'{column}_diff'].values
    f = Fitter(values,
               distributions=['gamma',
                              'lognorm',
                              "beta",
                              "burr",
                              "norm"])

    f.fit()
    print(f.summary())

    # Fit normal
    f_norm = Fitter(values, distributions=["norm"])
    f_norm.fit()
    print(f_norm.get_best(method='sumsquare_error'))


def two_sided_tests(df, column):
    # Test # 1 - distribution was drawn from a normal distribution
    shapiro = stats.shapiro(df[f'{column}_diff'])
    print(f'Shapiro test results {shapiro}')

    '''
    Examples for use are scores of the same set of student in different exams, or repeated sampling from the same units. 
    The test measures whether the average score differs significantly across samples (e.g. exams). 
    If we observe a large p-value, for example greater than 0.05 or 0.1 then we cannot reject the null hypothesis of identical average scores. 
    If the p-value is smaller than the threshold, e.g. 1%, 5% or 10%, then we reject the null hypothesis of equal averages. 
    Small p-values are associated with large t-statistics.
    '''
    ttest = stats.ttest_rel(df[column], df[f'{column}_after'])
    print(f'Ttest results: {ttest}')

    wilcoxon = stats.wilcoxon(df[column], df[f'{column}_after'])
    print(f'Wilcoxon test results: {wilcoxon}')


def transform_pai_z_data(df, rel=False):
    all_dfs = []
    for severity in [2, 3, 4]:
        for date_since in df.date_since.unique():
            df_derived = _unpack_pai_z(df, severity, date_since, rel)
            if df_derived is not None:
                all_dfs.append(df_derived)
    return pd.concat(all_dfs)


def _unpack_pai_z(df, severity, date_since, rel):
    if rel:
        pai_z = df[(df.burn_severity_median == severity) & (
            df.date_since == date_since)].pai_z_delta_np_rel.to_numpy()
    else:
        pai_z = df[(df.burn_severity_median == severity) & (
            df.date_since == date_since)].pai_z_delta_np_diff.to_numpy()

    if pai_z.shape[0] == 0:
        return

    unpacked = np.empty((pai_z.shape[0], pai_z[0].shape[0]))

    for i in range(pai_z.shape[0]):
        unpacked[i] = pai_z[i]

    new_df = pd.melt(pd.DataFrame(unpacked))
    new_df['severity'] = severity
    new_df['date_since'] = date_since
    return new_df
