import pandas as pd
import numpy as np
import sklearn.base
from sklearn.kernel_ridge import KernelRidge
from Model import Model
from econml.metalearners import SLearner as Test_SLearner
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

class SLearner(Model):
    def __init__(self, num_features: int, treatment_feature_name: str, target_feature_name: str,
                 interacted: bool = True, model=LinearRegression()):
        super().__init__(num_features, treatment_feature_name, target_feature_name)
        self.model = model
        self.interacted = interacted

    def fit(self, data: pd.DataFrame):
        results = data[self.target_name]
        original_features = data.drop(columns=[self.target_name])
        self.model.fit(self.__re_calculate_features(original_features), results)

    def reset(self):
        self.model = sklearn.base.clone(self.model)

    def calculate_ate(self, data: pd.DataFrame):
        original_features = data.drop(columns=[self.target_name])

        predictions_1 = self.model.predict(self.__re_calculate_features(original_features, force_treatment_value=1))
        predictions_0 = self.model.predict(self.__re_calculate_features(original_features, force_treatment_value=0))
        ATE = np.mean(predictions_1 - predictions_0)
        return ATE

    def __re_calculate_features(self, features: pd.DataFrame, force_treatment_value: int = None):
        features = features.copy()
        if force_treatment_value is not None:
            features[self.treatment_name] = force_treatment_value
        if self.interacted:
            features[[f"{self.treatment_name}_{column}" for column in features.columns]] = features.apply(
                lambda row: row * row[self.treatment_name], axis=1)
        return features


class BaselineSLearner(Model):
    def __init__(self, num_features: int, treatment_feature_name: str, target_feature_name: str):
        super(BaselineSLearner, self).__init__(num_features, treatment_feature_name, target_feature_name)

        self.model: Test_SLearner = None
        self.reset()

    def fit(self, data: pd.DataFrame):
        features = data.drop(columns=[self.treatment_name, self.target_name])
        self.model.fit(X=features, T=data[self.treatment_name], Y=data[self.target_name])

    def reset(self):
        self.model = Test_SLearner(overall_model=LinearRegression())

    def calculate_ate(self, data: pd.DataFrame):
        features = data.drop(columns=[self.treatment_name, self.target_name])
        return np.mean(self.model.effect(features))
