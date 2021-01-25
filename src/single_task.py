import numpy as np
from numpy.linalg import pinv
from numpy import identity as eye
from src.preprocessing import PreProcess
from time import time
from src.utilities import mae_clip
from sklearn.model_selection import KFold


def train_test_single_task(data, settings):
    # Training
    tt = time()

    """
    The goal is to merge all training/validation tasks and any training features from the test tasks and train a single model.
    If the number of training points on the test tasks is 0, then it's basically pure "transfer learning" from the training+ validation tasks onto the test tasks.
    """

    features_to_be_merged = data['tr_tasks_tr_features'] + data['val_tasks_tr_features'] + data['val_tasks_test_features'] + data['test_tasks_tr_features']
    labels_to_be_merged = data['tr_tasks_tr_labels'] + data['val_tasks_tr_labels'] + data['val_tasks_test_labels'] + data['test_tasks_tr_labels']
    x_merged = np.concatenate(features_to_be_merged)
    y_merged = np.concatenate(labels_to_be_merged)

    kf = KFold(n_splits=5)
    kf.get_n_splits(x_merged)
    preprocessing = PreProcess(threshold_scaling=True, standard_scaling=True, inside_ball_scaling=False, add_bias=True)

    best_performance = np.Inf
    best_param = None
    for regul_param in settings['regul_param_range']:
        curr_val_performances = []
        for train_index, test_index in kf.split(x_merged):
            x_tr, x_val = x_merged[train_index], x_merged[test_index]
            y_tr, y_val = y_merged[train_index], y_merged[test_index]

            x_tr, y_tr = preprocessing.transform(x_tr, y_tr, fit=True, multiple_tasks=False)
            x_val, y_val = preprocessing.transform(x_val, y_val, fit=False, multiple_tasks=False)

            model_itl = ITL(regul_param)
            model_itl.fit(x_tr, y_tr)

            val_predictions = model_itl.predict(x_val)
            val_performance = mae_clip(y_val, val_predictions)
            curr_val_performances.append(val_performance)
        average_val_performance = np.mean(curr_val_performances)
        if average_val_performance < best_performance:
            best_performance = average_val_performance
            best_param = regul_param

    x_merged, y_merged = preprocessing.transform(x_merged, y_merged, fit=True, multiple_tasks=False)

    model_itl = ITL(best_param)
    model_itl.fit(x_merged, y_merged)

    all_performances = []
    for task_idx in range(len(data['test_tasks_indexes'])):
        x_test, y_test = preprocessing.transform(data['test_tasks_test_features'][task_idx], data['test_tasks_test_labels'][task_idx], fit=False, multiple_tasks=False)

        # Testing
        test_predictions = model_itl.predict(x_test)
        all_performances.append(mae_clip(y_test, test_predictions))
    test_performance = np.mean(all_performances)
    print(f'{"Single task":12s} | test performance: {test_performance:12.5f} | {time() - tt:5.2f}sec')

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
