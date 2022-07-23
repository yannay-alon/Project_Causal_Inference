import pandas as pd
from sklearn.linear_model import LogisticRegression

from Model import Model


class IPW(Model):
    def __init__(self, num_features: int, treatment_feature_name: str, target_feature_name: str):
        super(IPW, self).__init__(num_features, treatment_feature_name, target_feature_name)

        self.model = LogisticRegression(C=1e4)

    def fit(self, data: pd.Dataframe):
        features = data.drop([self.treatment_name, self.target_name])
        labels = data[self.treatment_name]

        self.model.fit(features, labels)

    def predict(self, data: pd.DataFrame):
        features = data.drop([self.treatment_name, self.target_name])
        predicted_probabilities = self.model.predict_proba(features)[:, 1]

        return predicted_probabilities

    def reset(self):
        self.model = LogisticRegression(C=1e4)

    def calculate_ate(self, predictions: pd.DataFrame):
        pass
