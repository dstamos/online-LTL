import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from src.data_management_essex import concatenate_data
from numpy.linalg.linalg import norm


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


class PreProcess:
    def __init__(self, threshold_scaling, standard_scaling, inside_ball_scaling, add_bias=False):
        self.threshold_scaling = threshold_scaling
        self.standard_scaling = standard_scaling
        self.inside_ball_scaling = inside_ball_scaling
        self.add_bias = add_bias

    def transform(self, all_features, all_labels, training=False):
        concatenated_features, concatenated_labels, point_indexes_per_task = concatenate_data(all_features, all_labels)

        # These two scalers technically should be somehow applied before merging.
        # The reasoning is that metalearning is done in an online fashion, without reusing past data.
        if training is True:
            if self.threshold_scaling is True:
                outlier = ThressholdScaler()
                concatenated_features = outlier.fit_transform(concatenated_features)
                self.threshold_scaling = outlier

            if self.standard_scaling is True:
                sc = StandardScaler()
                concatenated_features = sc.fit_transform(concatenated_features)
                self.standard_scaling = sc
        else:
            if self.threshold_scaling is True:
                concatenated_features = self.threshold_scaling.transform(concatenated_features)

            if self.standard_scaling is True:
                concatenated_features = self.standard_scaling.transform(concatenated_features)

        if self.inside_ball_scaling is True:
            concatenated_features = concatenated_features / norm(concatenated_features, axis=0, keepdims=True)

        if self.add_bias is True:
            concatenated_features = np.concatenate((np.ones((len(concatenated_features), 1)), concatenated_features), 1)

        return concatenated_features, concatenated_labels, point_indexes_per_task
