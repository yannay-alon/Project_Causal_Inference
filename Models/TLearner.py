import pandas as pd
import numpy as np
from Model import Model
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

    def predict(self, data: pd.DataFrame):
        X_treated, _, _, _ = self.preprocess_data(data)
        return self.treated_model.predict(X_treated), self.untreated_model.predict(X_treated)

    def reset(self):
        self.treated_model = LinearRegression()
        self.untreated_model = LinearRegression()

    def calculate_ate(self, data: pd.DataFrame, treated_predictions: np.ndarray, untreated_predictions: np.ndarray):
        return 1 / len(treated_predictions) * sum(treated_predictions - untreated_predictions)

    def preprocess_data(self, data: pd.DataFrame):
        data_t_1 = data[data['T'] == 1]
        X_treated = data_t_1.filter(regex=("x_*"))
        y_treated = data_t_1['Y']

        data_t_0 = data[data['T'] == 0]
        X_untreated = data_t_0.filter(regex=("x_*"))
        y_untreated = data_t_0['Y']

        return X_treated, y_treated, X_untreated, y_untreated
