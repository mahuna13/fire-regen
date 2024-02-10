import pandas as pd
from src.counterfactuals.rf import train
from src.data.processing import gedi_raster_matching
from src.utils.logging_util import get_logger


logger = get_logger(__file__)


class RFTrainer():
    TERRAIN_FEATURES = ["slope", "elevation", "aspect"]

    def train(
            self,
            dep_var,
            features,
            train_df,
            test_df,
            log=False):
        m, to_train = train.train_rf(
            train_df[train_df[dep_var].notna()],
            dep_var,
            features,
            log=log,
            test_df=test_df[test_df[dep_var].notna()]
        )
        return m, to_train


class MonthlyLandsatRF(RFTrainer):
    def __init__(
        self,
        year: int,
        additional_features: list[str] = []
    ):
        self.year = year
        landsat_features = []
        for month in range(1, 13):
            for band in gedi_raster_matching.get_landsat_bands(year):
                landsat_features.append(f"{band}_{month}")

        self.features = \
            landsat_features + self.TERRAIN_FEATURES + additional_features

    def train(
            self,
            dep_var,
            train_df,
            test_df,
            log=False,
            augment=False):
        if augment:
            features, train_df, test_df = self._augment(train_df, test_df)
        else:
            features = [
                col for col in self.features if col in train_df.columns]

        logger.info(f"Training with features {features}.")
        return super().train(dep_var, features, train_df, test_df, log)

    def _augment(
            self,
            train_df,
            test_df):
        features = [col for col in self.features if col in train_df.columns]
        for band in gedi_raster_matching.get_landsat_bands(self.year):
            all_months = [col for col in train_df.columns
                          if col.startswith(band)]
            new_col = f"{band}_mean"
            train_df[new_col] = train_df[all_months].mean(axis=1)
            test_df[new_col] = test_df[all_months].mean(axis=1)
            features.append(new_col)

            new_col = f"{band}_max"
            train_df[new_col] = train_df[all_months].max(axis=1)
            test_df[new_col] = test_df[all_months].max(axis=1)
            features.append(new_col)

            new_col = f"{band}_min"
            train_df[new_col] = train_df[all_months].min(axis=1)
            test_df[new_col] = test_df[all_months].min(axis=1)
            features.append(new_col)

            new_col = f"{band}_coeff_v"
            train_df[new_col] = train_df[all_months].std(
                axis=1) / train_df[all_months].mean(axis=1)
            test_df[new_col] = test_df[all_months].std(
                axis=1) / test_df[all_months].mean(axis=1)
            features.append(new_col)

        return features, train_df, test_df


class LandsatTimeSeriesRF(RFTrainer):
    def __init__(
            self,
            years: list[str]):
        self.years = years
        landsat_features = []
        for year in years:
            for band in gedi_raster_matching.get_landsat_bands(year):
                landsat_features.append(f"{band}_{year}")
        self.features = landsat_features + self.TERRAIN_FEATURES

    def train(
            self,
            dep_var,
            train_df,
            test_df,
            log=False,
            augment=True):
        if augment:
            features, train_df, test_df = self._augment(train_df, test_df)
        else:
            features = self.features

        logger.info(f"Training with features {features}.")
        return super().train(dep_var, features, train_df, test_df, log)

    def _augment(
            self,
            train_df,
            test_df):
        aug_vars = []
        for band in gedi_raster_matching.get_landsat_bands(self.years[-1]):
            all_years = [f"{band}_{year}" for year in self.years]
            all_years = [col for col in all_years if col in train_df.columns]

            new_col = f"{band}_mean"
            train_df[new_col] = train_df[all_years].mean(axis=1)
            test_df[new_col] = test_df[all_years].mean(axis=1)
            aug_vars.append(new_col)

            new_col = f"{band}_std"
            train_df[new_col] = train_df[all_years].std(axis=1)
            test_df[new_col] = test_df[all_years].std(axis=1)
            aug_vars.append(new_col)

        features = self.features + aug_vars

        return features, train_df, test_df


class NDVITimeSeriesRF(RFTrainer):
    AUGMENTED_NDVI = ["min_ndvi", "max_ndvi", "std_ndvi", "mean_ndvi",
                      "median_ndvi"]

    def __init__(
            self,
            years: list[str]):
        self.years = years
        self.features = self.AUGMENTED_NDVI + self.TERRAIN_FEATURES + \
            [f"ndvi_{year}" for year in self.years]

    def augment_time_series_features(self, df):
        all_ndvi_cols = [f"ndvi_{year}" for year in self.years]
        df["min_ndvi"] = df[all_ndvi_cols].min(axis=1)
        df["max_ndvi"] = df[all_ndvi_cols].max(axis=1)
        df["std_ndvi"] = df[all_ndvi_cols].std(axis=1)
        df["mean_ndvi"] = df[all_ndvi_cols].mean(axis=1)
        df["median_ndvi"] = df[all_ndvi_cols].median(axis=1)
        return df

    def train(
            self,
            dep_var,
            train_df,
            test_df,
            log=False):
        train_df = self.augment_time_series_features(train_df)
        test_df = self.augment_time_series_features(test_df)
        return super().train(dep_var, self.features, train_df, test_df, log)


def rf_feat_importance(m, df):
    return pd.DataFrame({'cols': df.columns, 'imp': m.feature_importances_}
                        ).sort_values('imp', ascending=False)
