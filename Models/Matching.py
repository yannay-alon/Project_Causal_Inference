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

    def predict(self, data: pd.DataFrame):
        X_treated, _, _, _ = self.preprocess_data(data)
        return self.model.predict(X_treated)

    def reset(self):
        self.model = KNeighborsRegressor(n_neighbors=1)

    def calculate_ate(self, data: pd.DataFrame, predictions: np.ndarray):
        _, y_treated, _, _ = self.preprocess_data(data)
        return 1 / len(predictions) * sum(y_treated - predictions)

    def preprocess_data(self, data: pd.DataFrame):
        data_t_1 = data[data['T'] == 1]
        X_treated = data_t_1.filter(regex=("x_*"))
        y_treated = data_t_1['Y']

        data_t_0 = data[data['T'] == 0]
        X_untreated = data_t_0.filter(regex=("x_*"))
        y_untreated = data_t_0['Y']

        return X_treated, y_treated, X_untreated, y_untreated
