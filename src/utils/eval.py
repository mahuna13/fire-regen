import math

import numpy as np
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error, r2_score


def mse(y_true, y_pred):
    if y_true is None or y_pred is None:
        return None
    return mean_squared_error(y_true, y_pred)


def rmse(y_true, y_pred):
    if y_true is None or y_pred is None:
        return None
    return math.sqrt(mean_squared_error(y_true, y_pred))


def r_squared(y_true, y_pred):
    if y_true is None or y_pred is None:
        return None
    return r2_score(y_true, y_pred)


def rma_regression(x, y):
    """
    Calculates the Reduced Major Axis (RMA) regression line.

    Args:
      x: The independent variable.
      y: The dependent variable.

    Returns:
      A tuple of (slope, intercept).
    """

    # Calculate the ordinary least squares (OLS) regression line.
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # Calculate the covariance between x and y.
    cov = np.cov(x, y)[0, 1]

    # Calculate the variance of x.
    var_x = np.var(x)

    # Calculate the slope of the RMA regression line.
    rma_slope = cov / var_x

    # Calculate the intercept of the RMA regression line.
    rma_intercept = intercept - rma_slope * np.mean(x)

    return rma_slope, rma_intercept
