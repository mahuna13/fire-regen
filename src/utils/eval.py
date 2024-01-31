import math

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
