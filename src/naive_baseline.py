import numpy as np
from numpy.linalg import pinv
from numpy import identity as eye
from src.preprocessing import PreProcess
from time import time
from sklearn.metrics import mean_squared_error
from src.utilities import mae_clip


def train_test_itl(data, settings):
    # "Training"
    tt = time()
    all_performances = []
    for task_idx in range(len(data['test_tasks_indexes'])):
        # First merge, since there is no validation in this naive approach.
        y_tr = np.concatenate([data['test_tasks_tr_labels'][task_idx], data['test_tasks_val_labels'][task_idx]])
        y_test = data['test_tasks_val_features'][task_idx], data['test_tasks_val_labels'][task_idx], fit=False, multiple_tasks=False)

        best_performance = np.Inf
        for regul_param in settings['regul_param_range']:
            model_itl = ITL(regul_param)
            model_itl.fit(x_tr, y_tr)

            val_predictions = model_itl.predict(x_val)
            val_performance = mae_clip(y_val, val_predictions)
            if val_performance < best_performance:
                best_performance = val_performance
                best_model = model_itl

        # Testing
        test_predictions = best_model.predict(x_test)
        all_performances.append(mae_clip(y_test, test_predictions))
    test_performance = np.median(all_performances)
    print(f'ITL | test performance: {test_performance:12.5f} | {time() - tt:5.2f}sec')

    return test_performance


class ITL:
    def __init__(self, regularization_parameter=1e-2):
        self.regularization_parameter = regularization_parameter
        self.weight_vector = None

    def fit(self, features, labels):
        dims = features.shape[1]

        weight_vector = pinv(features.T @ features + self.regularization_parameter * eye(dims)) @ features.T @ labels
        self.weight_vector = weight_vector

    def predict(self, features):
        pred = np.matmul(features, self.weight_vector)
        return pred
