import pandas as pd
import numpy as np
from Model import Model
from sklearn.neighbors import KNeighborsRegressor


class Matching(Model):
    def __init__(self, num_features: int, treatment_feature_name: str, target_feature_name: str,
                 model=KNeighborsRegressor(n_neighbors=1)):
        super().__init__(num_features, treatment_feature_name, target_feature_name)
        self.model = model

    def fit(self, data: pd.DataFrame):
        _, _, X_untreated, y_untreated = self.preprocess_data(data)
        self.model.fit(X_untreated, y_untreated)

    def reset(self):
        self.model = KNeighborsRegressor(n_neighbors=1)

    def calculate_ate(self, data: pd.DataFrame):
        X_treated, y_treated, _, _ = self.preprocess_data(data)
        predictions = self.model.predict(X_treated)
        return 1 / len(predictions) * sum(y_treated - predictions)

    def preprocess_data(self, data: pd.DataFrame):
        data_t_1 = data[data[self.treatment_name] == 1]
        X_treated = data_t_1.drop(columns=[self.treatment_name, self.target_name])
        y_treated = data_t_1[self.target_name]

        data_t_0 = data[data[self.treatment_name] == 0]
        X_untreated = data_t_0.drop(columns=[self.treatment_name, self.target_name])
        y_untreated = data_t_0[self.target_name]

        return X_treated, y_treated, X_untreated, y_untreated
