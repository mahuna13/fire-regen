import matplotlib.pyplot as plt
from scipy.stats import linregress, gaussian_kde
import numpy as np
import pandas as pd


def plot_pdf(df: pd.DataFrame, x_col: str, y_col: str, x_label: str,
             y_label: str, y_lim: tuple):
    n = len(df)
    print(n)

    x = df[x_col].values
    y = df[y_col].values

    # Calculate the point density distribution per age group
    x_values = np.unique(x)
    z = np.zeros(y.shape)
    for i in x_values:
        z[x == i] = gaussian_kde(y[x == i])(y[x == i])

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    # limit to only the points with high enough density
    # x = x[z > 0.005]
    # y = y[z > 0.005]
    # z = z[z > 0.005]

    fig, ax = plt.subplots(1, 1, figsize=(25, 10))
    im = ax.scatter(x, y, c=z, s=500, cmap='PuRd')
    ax.set_ylim(y_lim)
    ax.set_xticks(x_values[::2])
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlabel(x_label, size=15)
    ax.set_ylabel(y_label, size=15)
    plt.colorbar(im)

    plt.show()
