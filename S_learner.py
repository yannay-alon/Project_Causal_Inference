import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.base
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from causalinference import CausalModel
from sklearn import metrics, calibration
from sklearn.linear_model import LogisticRegression, LinearRegression
from Model import Model


class S_learner(Model):
    def __init__(self, num_features: int, treatment_feature_name: str, target_feature_name: str,
                 interacted: bool = True, model=KernelRidge(kernel="poly", degree=3)):
        super().__init__(num_features, treatment_feature_name, target_feature_name)
        self.model = model
        self.interacted = interacted

    def fit(self, data: pd.DataFrame):
        results = data[self.target_name]
        original_features = data.drop(columns=[self.target_name])
        self.model.fit(self.__re_calculate_features(original_features), results)

    def predict(self, data: pd.DataFrame):
        original_features = data.drop(columns=[self.target_name])
        return self.model.predict(self.__re_calculate_features(original_features))

    def reset(self):
        self.model = sklearn.base.clone(self.model)

    def calculate_ate(self, data: pd.DataFrame, predictions: np.ndarray):
        original_features = data.drop(columns=[self.target_name])
        treatments = data[self.treatment_name]
        results = data[self.target_name]

        predictions_1 = self.model.predict(self.__re_calculate_features(original_features, force_treatment_value=1))
        predictions_0 = self.model.predict(self.__re_calculate_features(original_features, force_treatment_value=0))
        ATE = np.mean(predictions_1 - predictions_0)
        return ATE

    def __re_calculate_features(self, features: pd.DataFrame, force_treatment_value: int = None):
        features = features.copy()
        if force_treatment_value is not None:
            features["T"] = force_treatment_value
        if self.interacted:
            features[[f"T_{column}" for column in features.columns]] = features.apply(lambda row: row * row['T'],
                                                                                      axis=1)
        return features
