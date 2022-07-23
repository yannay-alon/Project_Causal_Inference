import pandas as pd


class Model:
    def __init__(self, num_features: int, treatment_feature_name: str, target_feature_name: str):
        self.num_features = num_features
        self.treatment_name = treatment_feature_name
        self.target_name = target_feature_name

    def fit(self, data: pd.Dataframe):
        raise NotImplementedError

    def predict(self, data: pd.DataFrame):
        raise NotImplementedError
