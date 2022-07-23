import numpy as np
import pandas as pd


class Model:
    def __init__(self, num_features: int, treatment_feature_name: str, target_feature_name: str):
        self.num_features = num_features
        self.treatment_name = treatment_feature_name
        self.target_name = target_feature_name

    def fit(self, data: pd.DataFrame):
        raise NotImplementedError

    def predict(self, data: pd.DataFrame):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def calculate_ate(self, data: pd.DataFrame, predictions: np.ndarray):
        raise NotImplementedError
