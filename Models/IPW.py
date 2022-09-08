import numpy as np
import pandas as pd
import sklearn.base
from causallib.estimation import IPW as BASELINE_IPW
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from Model import Model


class IPW(Model):
    def __init__(self, num_features: int, treatment_feature_name: str, target_feature_name: str):
        super(IPW, self).__init__(num_features, treatment_feature_name, target_feature_name)

        self.model = SVC(probability=True)

        self.reset()

    def fit(self, data: pd.DataFrame):
        features = data.drop(columns=[self.treatment_name, self.target_name])
        labels = data[self.treatment_name]
        self.model.fit(features, labels)

    def reset(self):
        self.model = sklearn.base.clone(self.model)

    def calculate_ate(self, data: pd.DataFrame):
        features = data.drop(columns=[self.treatment_name, self.target_name])
        predictions = self.model.predict_proba(features)
        weights = (data[self.treatment_name] - predictions[:, 1]) / (predictions[:, 1] * predictions[:, 0])
        return np.mean(weights * data[self.target_name])


class BaselineIPW(Model):
    def __init__(self, num_features: int, treatment_feature_name: str, target_feature_name: str):
        super(BaselineIPW, self).__init__(num_features, treatment_feature_name, target_feature_name)

        self.model: BASELINE_IPW = None

        self.reset()

    def fit(self, data: pd.DataFrame):
        features = data.drop(columns=[self.treatment_name, self.target_name])
        labels = data[self.treatment_name]

        self.model.fit(features, labels)

    def reset(self):
        self.model = BASELINE_IPW(SVC(probability=True))

    def calculate_ate(self, data: pd.DataFrame):
        features = data.drop(columns=[self.treatment_name, self.target_name])

        treatments = data[self.treatment_name]
        results = data[self.target_name]

        potential_outcomes = self.model.estimate_population_outcome(features, treatments, results)
        return self.model.estimate_effect(potential_outcomes[1], potential_outcomes[0]).item()
