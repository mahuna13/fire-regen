import argparse

import geopandas as gpd
import numpy as np
from fastai.tabular.all import load_pickle, save_pickle
from sklearn.neighbors import BallTree
from src.counterfactuals.counterfactual import CounterfactualGenerator
from src.utils.logging_util import get_logger

logger = get_logger(__file__)

MATCHING_COLUMNS = ["elevation", "slope", "ndvi_2019", "ndvi_2015",
                    "ndvi_2010"]
OUTCOME_VARS = ["pai", "rh_98", "rh_70", "rh_50", "cover", "ndvi_2020",
                "ndvi_2021", "ndvi_2022"]


class MatchingGenerator(CounterfactualGenerator):
    def __init__(
        self,
        outcome_vars: list[str],
        k_neighbors: int
    ):
        self.outcome_vars = outcome_vars
        self.k_n = k_neighbors

    def generate(
            self,
            treated_df: gpd.GeoDataFrame,
            untreated_df: gpd.GeoDataFrame):
        treated_idx, matches_idx = self.find_matches(treated_df, untreated_df)

        treated_matches = treated_df.loc[treated_idx]
        untreated_matches = untreated_df.loc[matches_idx]

        # Calculate std diff.
        for var in MATCHING_COLUMNS:
            std_diff = abs(
                treated_matches[var].mean() - untreated_matches[var].mean()
            ) / treated_matches[var].std()
            logger.info(f"Std Diff for {var}: {std_diff}")

        # Calculate counterfactuals.
        for outcome_var in OUTCOME_VARS:
            treated_matches[f"{outcome_var}_cf"] = \
                untreated_matches[outcome_var].to_numpy()

        print(treated_matches)
        return treated_matches

    def find_matches(
            self,
            treated: gpd.GeoDataFrame,
            untreated: gpd.GeoDataFrame):
        logger.info("Find matches.")

        # For each sample in treated
        left_df = treated[MATCHING_COLUMNS].dropna()
        right_df = untreated[MATCHING_COLUMNS].dropna()

        left = left_df.to_numpy()
        right = right_df.to_numpy()

        logger.info("Calculate covariate matrix.")
        V = np.cov(right, rowvar=False)

        logger.info("Construct search tree for mahalanobis distance.")
        tree = BallTree(right, leaf_size=1, metric='mahalanobis', V=V)

        logger.info('Find nearest neighbors.')
        distances, indeces = tree.query(left, k=self.k_n)

        logger.info('Return indeces of nearest matches.')
        matches = right_df.iloc[indeces[:, 0]]
        return left_df.index, matches.index

    def save_pickle_file(
        self,
        df,
        name: str,
        save_path: str
    ):
        FILE_NAME = f"matching_{name}_{self.k_n}.pkl"
        file_path = f"{save_path}/{FILE_NAME}"
        logger.info(
            f"Save file: {file_path}.")
        save_pickle(file_path, df)


def main(
    name: str,
    treated: gpd.GeoDataFrame,
    untreated: gpd.GeoDataFrame,
    k_neighbors: int,
    save_path: str
):
    mg = MatchingGenerator(OUTCOME_VARS, k_neighbors)
    treated_counterfactuals = mg.generate(treated, untreated)

    if save_path is not None:
        mg.save_pickle_file(treated_counterfactuals, name, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script to generate counterfactuals based on the matching \
        method.")

    parser.add_argument(
        "-n",
        "--name",
        help="Name of the run, for easier referencing.",
        type=str,
    )

    parser.add_argument(
        "-t",
        "--treated",
        help="The path to the pkl file with dataframe for treated samples.",
        type=str,
    )

    parser.add_argument(
        "-u",
        "--untreated",
        help="The path to the pkl file with dataframe for untreated samples.",
        type=str,
    )

    parser.add_argument(
        "-k",
        "--k_neighbors",
        help="Number of nearest neighbors to use.",
        type=int,
        default=100
    )

    parser.add_argument(
        "-p",
        "--path",
        help="Path to the folder where the result should be saved.",
        type=str,
    )

    args = parser.parse_args()
    main(
        name=args.name,
        treated=load_pickle(args.treated),
        untreated=load_pickle(args.untreated),
        k_neighbors=args.k_neighbors,
        save_path=args.path
    )
