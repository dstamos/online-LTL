import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ThressholdScaler(BaseEstimator, TransformerMixin):
    def __init__(self, qoffset=1.5):
        self.qoffset = qoffset
        self._pthr = 0
        self._nthr = 0

    def fit(self, X, y=None):
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        self._nthr = q1 - self.qoffset*(q3 - q1)
        self._pthr = q3 + self.qoffset*(q3 - q1)
        return self

    def transform(self, X, y=None):
        for i in range(X.shape[1]):
            X[:, i] = np.clip(X[:, i], self._nthr[i], self._pthr[i])
        return X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)