import pandas as pd
import numpy as np
from Model import Model
from econml.metalearners import TLearner as Test_TLearner
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


class TLearner(Model):
    def __init__(self, num_features: int, treatment_feature_name: str, target_feature_name: str,
                 treated_model=LinearRegression(), untreated_model=LinearRegression()):
        super().__init__(num_features, treatment_feature_name, target_feature_name)
        self.treated_model = treated_model
        self.untreated_model = untreated_model

    def fit(self, data: pd.DataFrame):
        X_treated, y_treated, X_untreated, y_untreated = self.preprocess_data(data)

        self.treated_model.fit(X_treated, y_treated)
        self.untreated_model.fit(X_untreated, y_untreated)

    def reset(self):
        self.treated_model = LinearRegression()
        self.untreated_model = LinearRegression()

    def calculate_ate(self, data: pd.DataFrame):
        X_treated, _, _, _ = self.preprocess_data(data)
        treated_predictions = self.treated_model.predict(X_treated)
        untreated_predictions = self.untreated_model.predict(X_treated)
        return 1 / len(treated_predictions) * sum(treated_predictions - untreated_predictions)

    def preprocess_data(self, data: pd.DataFrame):
        data_t_1 = data[data[self.treatment_name] == 1]
        X_treated = data_t_1.drop(columns=[self.treatment_name, self.target_name])
        y_treated = data_t_1[self.target_name]

        data_t_0 = data[data[self.treatment_name] == 0]
        X_untreated = data_t_0.drop(columns=[self.treatment_name, self.target_name])
        y_untreated = data_t_0[self.target_name]

        return X_treated, y_treated, X_untreated, y_untreated


class BaselineTLearner(Model):

    def __init__(self, num_features: int, treatment_feature_name: str, target_feature_name: str):
        super(BaselineTLearner, self).__init__(num_features, treatment_feature_name, target_feature_name)

        self.model: Test_TLearner = None
        self.reset()

    def fit(self, data: pd.DataFrame):
        features = data.drop(columns=[self.treatment_name, self.target_name])
        self.model.fit(X=features, T=data[self.treatment_name], Y=data[self.target_name])

    def reset(self):
        self.model = Test_TLearner(models=LinearRegression())

    def calculate_ate(self, data: pd.DataFrame):
        features = data.drop(columns=[self.treatment_name, self.target_name])
        return np.mean(self.model.effect(features))
