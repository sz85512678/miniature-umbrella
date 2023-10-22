import numpy as np
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge

import lightgbm as lgb
from sklearn.linear_model import ElasticNet

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')


class MyDummyClassifier(BaseEstimator):
    def __init__(self, constant=0):
        self.model = DummyClassifier()

    def fit(self, X, y=None):
        self.model.fit(X, y)

    def predict(self, x):
        return self.model.predict(x)

    def importance(self):
        return np.array(0)


