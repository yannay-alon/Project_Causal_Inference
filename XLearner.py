import numpy as np
import pandas as pd
from typing import List
from causallib.estimation import Standardization, StratifiedStandardization, XLearner as Test_XLearner
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from Model import Model

__all__ = ["XLearner", "BaselineXLearner"]


class XLearner(Model):
    def __init__(self, num_features: int, treatment_feature_name: str, target_feature_name: str):
        super(XLearner, self).__init__(num_features, treatment_feature_name, target_feature_name)

        self.propensity_model: GradientBoostingClassifier = GradientBoostingClassifier()
        self.first_stage: List[GradientBoostingClassifier] = None
        self.second_stage: List[GradientBoostingRegressor] = None
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

    def predict(self, data: pd.DataFrame):
        features = data.drop(columns=[self.treatment_name, self.target_name])
        return self.propensity_model.predict_proba(features)[:, 1]

    def reset(self):
        self.first_stage = [
            GradientBoostingClassifier(max_depth=2),
            GradientBoostingClassifier(max_depth=2)
        ]

        self.propensity_model = GradientBoostingClassifier()

        self.second_stage = [
            GradientBoostingRegressor(max_depth=2),
            GradientBoostingRegressor(max_depth=2)
        ]

    def calculate_ate(self, data: pd.DataFrame, predictions: np.ndarray):
        features = data.drop(columns=[self.treatment_name, self.target_name])

        propensity_scores = predictions
        predictions = [self.second_stage[index].predict(features) for index in range(len(self.treatment_values))]
        return np.mean(propensity_scores * predictions[0] + (1 - propensity_scores) * predictions[1]).item()


class BaselineXLearner(Model):
    def __init__(self, num_features: int, treatment_feature_name: str, target_feature_name: str):
        super(BaselineXLearner, self).__init__(num_features, treatment_feature_name, target_feature_name)

        self.model: Test_XLearner = None
        self.reset()

    def fit(self, data: pd.DataFrame):
        features = data.drop(columns=[self.treatment_name, self.target_name])
        self.model.fit(features, data[self.treatment_name], data[self.target_name])

    def predict(self, data: pd.DataFrame):
        features = data.drop(columns=[self.treatment_name, self.target_name])
        return self.model.treatment_model.predict_proba(features)[:, 1]

    def reset(self):
        self.model = Test_XLearner(
            outcome_model=StratifiedStandardization(GradientBoostingRegressor()),
            effect_model=Standardization(GradientBoostingRegressor()),
            effect_types="diff"
        )

    def calculate_ate(self, data: pd.DataFrame, predictions: np.ndarray):
        features = data.drop(columns=[self.treatment_name, self.target_name])
        return self.model.estimate_effect(features, data[self.treatment_name], agg="population").item()