import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from Model import Model


class IPW(Model):
    def __init__(self, num_features: int, treatment_feature_name: str, target_feature_name: str):
        super(IPW, self).__init__(num_features, treatment_feature_name, target_feature_name)

        self.model = None
        self.reset()

    def fit(self, data: pd.DataFrame):
        features = data.drop(columns=[self.treatment_name, self.target_name])
        labels = data[self.treatment_name]

        self.model.fit(features, labels)

    def predict(self, data: pd.DataFrame):
        features = data.drop(columns=[self.treatment_name, self.target_name])
        predicted_probabilities = self.model.predict_proba(features)[:, 1]

        return predicted_probabilities

    def reset(self):
        self.model = LogisticRegression(C=1e6, max_iter=1e8)

    def calculate_ate(self, data: pd.DataFrame, predictions: np.ndarray):
        weights = (data[self.treatment_name] - predictions) / (predictions * (1 - predictions))
        return np.mean(weights * data[self.target_name]).item()
