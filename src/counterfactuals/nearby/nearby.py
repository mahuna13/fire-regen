import argparse

import geopandas as gpd
import numpy as np
from fastai.tabular.all import load_pickle, save_pickle
from src.counterfactuals.counterfactual import CounterfactualGenerator
from src.data.processing import k_nn
from src.utils.logging_util import get_logger

logger = get_logger(__file__)

OUTCOME_VARS = ["pai", "ndvi", "rh_98", "rh_70", "rh_50", "cover"]


class NearbyGenerator(CounterfactualGenerator):
    def __init__(
            self,
            outcome_vars: list[str],
            k_neighbors: int,
            estimator: str = "mean"):
        self.outcome_vars = outcome_vars
        self.k_neighbors = k_neighbors
        self.estimator = estimator

    def generate(
            self,
            treated_df: gpd.GeoDataFrame,
            untreated_df: gpd.GeoDataFrame):

        logger.info(
            f"Calling nearest neighbor to find {self.k_neighbors} nn.")
        # For each treated sample, find the nearby untreated.
        nn_indeces, nn_distances = k_nn.nearest_neighbors(
            treated_df, untreated_df, self.k_neighbors)

        for outcome_var in self.outcome_vars:
            logger.info(f"Calculating counterfactual for {outcome_var}")
            treated_df[f"{outcome_var}_cf"] = np.apply_along_axis(
                lambda x: self._estimate(untreated_df.iloc[x][outcome_var]),
                1,  # axis
                nn_indeces
            )

        return treated_df

    def _estimate(
        self,
        df
    ):
        if self.estimator == "mean":
            return df.mean()
        elif self.estimator == "median":
            return df.median()
        else:
            return None

    def save_pickle_file(
        self,
        df,
        name: str,
        save_path: str
    ):
        FILE_NAME = f"nearby_{name}_{self.k_neighbors}_{self.estimator}.pkl"
        file_path = f"{save_path}/{FILE_NAME}"
        logger.info(
            f"Save file: {file_path}.")
        save_pickle(file_path, df)


def main(
    name: str,
    treated: gpd.GeoDataFrame,
    untreated: gpd.GeoDataFrame,
    k_neighbors: int,
    estimator: str,
    save_path: str
):
    ng = NearbyGenerator(OUTCOME_VARS, k_neighbors, estimator)
    treated_counterfactuals = ng.generate(treated, untreated)

    if save_path is not None:
        ng.save_pickle_file(treated_counterfactuals, name, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script to generate counterfactuals based on the nearby \
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
        "-e",
        "--estimator",
        help="The name of the estimator to use - mean or median.",
        type=str,
        default="mean"
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
        estimator=args.estimator,
        save_path=args.path
    )
