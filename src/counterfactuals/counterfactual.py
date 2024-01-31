import pandas as pd


class CounterfactualGenerator():
    def __init__(
            self,
            outcome_vars: list[str]):
        self.outcome_vars = outcome_vars

    def train(
            self,
            train_df: pd.DataFrame):
        pass

    def generate(
            self,
            input_df: pd.DataFrame):
        pass
