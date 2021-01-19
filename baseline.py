from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge as lr
from src.preprocessing import ThressholdScaler, BallScaling
from src.loadData import load_data_essex, load_data_Chris
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict as cv
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class BaselineEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.pred = 0

    def fit(self, X, y=None):
        self.pred = np.mean(y)
        return self

    def predict(self, X):
        return np.zeros((np.shape(X)[0],1)) + self.pred

if __name__ == "__main__":
    all_features, all_labels = load_data_Chris()
    nSubj = len(all_features)

    basePipe = Pipeline([('Th', ThressholdScaler()), ('Sc', StandardScaler()), ('Bs', BallScaling()), ('pred', BaselineEstimator())])
    linePipe = Pipeline([('Th', ThressholdScaler()), ('Sc', StandardScaler()), ('Bs', BallScaling()), ('pred', lr())])

    baseMAESS = np.zeros((nSubj, 1))
    lineMAESS = np.zeros((nSubj, 1))

    for i in range(nSubj):
        pred = cv(basePipe, all_features[i], all_labels[i], cv=5, n_jobs=-1)
        baseMAESS[i] = np.median(np.abs(pred - all_labels[i]))
        pred = cv(linePipe, all_features[i], all_labels[i], cv=5, n_jobs=-1)
        lineMAESS[i] = np.median(np.abs(pred - all_labels[i]))

    baseMAEMS = np.zeros((nSubj, 1))
    lineMAEMS = np.zeros((nSubj, 1))
    for i in range(nSubj):
        x_tr, y_tr =[], []
        for j in range(nSubj):
            if i != j:
                x_tr.append(all_features[j])
                y_tr.append(all_labels[j])
        x_tr = np.concatenate(x_tr)
        y_tr = np.concatenate(y_tr)
        basePipe.fit(x_tr, y_tr)
        pred = basePipe.predict(all_features[i])
        baseMAEMS[i] = np.median(np.abs(pred - all_labels[i]))
        linePipe.fit(x_tr, y_tr)
        pred = linePipe.predict(all_features[i])
        lineMAEMS[i] = np.median(np.abs(pred - all_labels[i]))
