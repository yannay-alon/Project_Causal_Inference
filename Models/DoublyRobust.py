import pandas as pd
import numpy as np
import sklearn.base
from Model import Model
from causallib.estimation import PropensityFeatureStandardization, StratifiedStandardization, OverlapWeights
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC



class DoublyRobust(Model):
    def __init__(self, num_features: int, treatment_feature_name: str, target_feature_name: str,
                 propensity_model=SVC(probability=True),
                 treated_model=SVC(probability=True),
                 controlled_model=SVC(probability=True)):
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

    def reset(self):
        self.propensity_model = sklearn.base.clone(self.propensity_model)
        self.treated_model = sklearn.base.clone(self.treated_model)
        self.controlled_model = sklearn.base.clone(self.controlled_model)

    def calculate_ate(self, data: pd.DataFrame):
        x = data.columns.drop([self.treatment_name, self.target_name])
        t = self.treatment_name
        y = self.target_name
        ps = self.propensity_model.predict_proba(data.drop(columns=[self.treatment_name, self.target_name]))
        mu0 = self.treated_model.predict(data[x])
        mu1 = self.controlled_model.predict(data[x])

        treated_effect = np.mean(data[t] * (data[y] - mu1) / ps[:, 1] + mu1)
        control_effect = np.mean((1 - data[t]) * (data[y] - mu0) / ps[:, 0] + mu0)
        return treated_effect - control_effect


class BaselineDoublyRobust(Model):
    def __init__(self, num_features: int, treatment_feature_name: str, target_feature_name: str):
        super(BaselineDoublyRobust, self).__init__(num_features, treatment_feature_name, target_feature_name)

        self.model: PropensityFeatureStandardization = None
        self.reset()

    def fit(self, data: pd.DataFrame):
        features = data.drop(columns=[self.treatment_name, self.target_name])
        self.model.fit(features, data[self.treatment_name], data[self.target_name])

    def reset(self):
        self.model = PropensityFeatureStandardization(
            outcome_model=StratifiedStandardization(SVC(probability=True)),
            weight_model=OverlapWeights(SVC(probability=True))
        )

    def calculate_ate(self, data: pd.DataFrame):
        features = data.drop(columns=[self.treatment_name, self.target_name])
        treatment_outcomes = self.model.estimate_population_outcome(X=features, a=data[self.treatment_name])
        return treatment_outcomes[1] - treatment_outcomes[0]
