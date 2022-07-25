import numpy as np
import pandas as pd
from causallib.estimation import IPW as TEST_IPW
from sklearn.ensemble import GradientBoostingClassifier

from Model import Model

__all__ = ["IPW", "BaselineIPW"]


class IPW(Model):
    def __init__(self, num_features: int, treatment_feature_name: str, target_feature_name: str):
        super(IPW, self).__init__(num_features, treatment_feature_name, target_feature_name)

        self.model: GradientBoostingClassifier = None

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
        self.model = GradientBoostingClassifier()

    def calculate_ate(self, data: pd.DataFrame, predictions: np.ndarray):
        weights = (data[self.treatment_name] - predictions) / (predictions * (1 - predictions))
        return np.mean(weights * data[self.target_name]).item()


class BaselineIPW(Model):
    def __init__(self, num_features: int, treatment_feature_name: str, target_feature_name: str):
        super(BaselineIPW, self).__init__(num_features, treatment_feature_name, target_feature_name)

        self.model: TEST_IPW = None

        self.reset()

    def fit(self, data: pd.DataFrame):
        features = data.drop(columns=[self.treatment_name, self.target_name])
        labels = data[self.treatment_name]

        self.model.fit(features, labels)

    def predict(self, data: pd.DataFrame):
        features = data.drop(columns=[self.treatment_name, self.target_name])

        predicted_probabilities = self.model.learner.predict_proba(features)[:, 1]
        return predicted_probabilities

    def reset(self):
        self.model = TEST_IPW(GradientBoostingClassifier())

    def calculate_ate(self, data: pd.DataFrame, predictions: np.ndarray):
        features = data.drop(columns=[self.treatment_name, self.target_name])

        treatments = data[self.treatment_name]
        results = data[self.target_name]

        potential_outcomes = self.model.estimate_population_outcome(features, treatments, results)
        return self.model.estimate_effect(potential_outcomes[1], potential_outcomes[0])
