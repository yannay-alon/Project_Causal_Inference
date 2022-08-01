import pandas as pd
import numpy as np
import sklearn.base
from sklearn.linear_model import LogisticRegression, LinearRegression
from Model import Model


class DoublyRobust(Model):
    def __init__(self, num_features: int, treatment_feature_name: str, target_feature_name: str,
                 propensity_model=LogisticRegression(max_iter=1e6),
                 treated_model=LinearRegression(), controlled_model=LinearRegression()):
        super().__init__(num_features, treatment_feature_name, target_feature_name)
        self.propensity_model = propensity_model
        self.treated_model = treated_model
        self.controlled_model = controlled_model

    def fit(self, data: pd.DataFrame):
        original_features = data.columns.drop([self.treatment_name, self.target_name])

        self.propensity_model.fit(data[original_features], data[self.treatment_name])
        self.treated_model.fit(data.query(f"{self.treatment_name}==0")[original_features],
                               data.query(f"{self.treatment_name}==0")[self.target_name])
        self.controlled_model.fit(data.query(f"{self.treatment_name}==1")[original_features],
                                  data.query(f"{self.treatment_name}==1")[self.target_name])

    def predict(self, data: pd.DataFrame):
        ps = self.propensity_model.predict_proba(data.drop([self.treatment_name, self.target_name]))[:, 1]
        return ps

    def reset(self):
        self.propensity_model = sklearn.base.clone(self.propensity_model)
        self.treated_model = sklearn.base.clone(self.treated_model)
        self.controlled_model = sklearn.base.clone(self.controlled_model)

    def calculate_ate(self, data: pd.DataFrame, predictions: np.ndarray):
        x = data.columns.drop([self.treatment_name, self.target_name])
        t = self.treatment_name
        y = self.target_name
        ps = predictions  # propensities
        mu0 = self.treated_model.predict(data[x])
        mu1 = self.controlled_model.predict(data[x])
        return np.mean(data[t] * (data[y] - mu1) / ps + mu1) - np.mean((1 - data[t]) * (data[y] - mu0) / (1 - ps) + mu0)
