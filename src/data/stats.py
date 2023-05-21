from scipy import stats
import numpy as np
import pandas as pd


def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    filtered = df.query(
        f'(@Q1 - 1.5 * @IQR) <= {column} <= (@Q3 + 1.5 * @IQR)')
    return filtered


def AB_test(test: pd.Series, control: pd.Series, confidence=0.95, h0=0):
    # Remove outliers

    mu1, mu2 = test.mean(), control.mean()
    se1, se2 = test.std() / np.sqrt(len(test)), control.std() / np.sqrt(len(control))
    diff = mu1 - mu2
    se_diff = np.sqrt(test.var()/len(test) + control.var()/len(control))
    z_stats = (diff-h0)/se_diff
    p_value = stats.norm.cdf(z_stats)

    def critial(se):
        return -se*stats.norm.ppf((1 - confidence)/2)
    print(f"Test Mean {mu1}, Test STD {test.std()}")
    print(f"Control Mean {mu2}, Control STD {control.std()}")
    print(f"Test {confidence*100}% CI: {mu1} +- {critial(se1)}")
    print(f"Control {confidence*100}% CI: {mu2} +- {critial(se2)}")
    print(f"Test-Control {confidence*100}% CI: {diff} +- {critial(se_diff)}")
    print(f"Z Statistic {z_stats}")
    print(f"P-Value {p_value}")
