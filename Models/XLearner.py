import numpy as np
import pandas as pd
from typing import List
from causallib.estimation import Standardization, StratifiedStandardization, XLearner as Test_XLearner
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC



from Model import Model

__all__ = ["XLearner", "BaselineXLearner"]


class XLearner(Model):
    def __init__(self, num_features: int, treatment_feature_name: str, target_feature_name: str):
        super(XLearner, self).__init__(num_features, treatment_feature_name, target_feature_name)

        self.propensity_model: SVC = SVC(probability=True)
        self.first_stage: List[SVC] = None
        self.second_stage: List[LinearRegression] = None
        self.reset()

        self.treatment_values = [0, 1]  # Only supports binary treatments

    def fit(self, data: pd.DataFrame):
        treatment_masks = [data[self.treatment_name] == i for i in self.treatment_values]
        features = data.drop(columns=[self.treatment_name, self.target_name])

        self.propensity_model.fit(features, data[self.treatment_name])

        predictions = []
        for index, treatment_mask in enumerate(treatment_masks):
            self.first_stage[index].fit(features[treatment_mask], data[self.target_name][treatment_mask])
            predictions.append(self.first_stage[index].predict_proba(features)[:, 1])

        imputed_treatment_effects = np.where(treatment_masks[1],
                                             data[self.target_name] - predictions[0],
                                             predictions[1] - data[self.target_name])

        for index, treatment_mask in enumerate(treatment_masks):
            self.second_stage[index].fit(features[treatment_mask], imputed_treatment_effects[treatment_mask])

    def reset(self):
        self.first_stage = [
            SVC(probability=True),
            SVC(probability=True)
        ]

        self.propensity_model = SVC(probability=True)

        self.second_stage = [
            LinearRegression(),
            LinearRegression()
        ]

    def calculate_ate(self, data: pd.DataFrame):
        features = data.drop(columns=[self.treatment_name, self.target_name])

        predictions_scores = self.propensity_model.predict_proba(features)
        predictions = [self.second_stage[index].predict(features) for index in range(len(self.treatment_values))]
        return np.mean(predictions_scores[:, 1] * predictions[0] + predictions_scores[:, 0] * predictions[1]).item()


class BaselineXLearner(Model):
    def __init__(self, num_features: int, treatment_feature_name: str, target_feature_name: str):
        super(BaselineXLearner, self).__init__(num_features, treatment_feature_name, target_feature_name)

        self.model: Test_XLearner = None
        self.reset()

    def fit(self, data: pd.DataFrame):
        features = data.drop(columns=[self.treatment_name, self.target_name])
        self.model.fit(features, data[self.treatment_name], data[self.target_name])

    def reset(self):
        self.model = Test_XLearner(
            outcome_model=StratifiedStandardization(LinearRegression()),
            effect_model=Standardization(LinearRegression()),
            effect_types="diff"
        )

    def calculate_ate(self, data: pd.DataFrame):
        features = data.drop(columns=[self.treatment_name, self.target_name])
        return self.model.estimate_effect(features, data[self.treatment_name], agg="population").item()
