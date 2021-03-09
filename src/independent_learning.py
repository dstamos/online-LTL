import warnings
import numpy as np
from scipy.linalg import lstsq
from src.preprocessing import PreProcess
from time import time
from src.utilities import evaluation_methods
from sklearn.model_selection import KFold, ShuffleSplit


def train_test_itl(data, settings):
    # Training
    tt = time()
    all_performances = []
    all_predictions = []
    all_weights = []
    for task_idx in range(len(data['test_tasks_indexes'])):
        x = data['test_tasks_tr_features'][task_idx]
        y = data['test_tasks_tr_labels'][task_idx]
        corr = data['test_tasks_tr_corr'][task_idx]

        cv_splits = 5
        if len(y) < cv_splits:
            # In the case we don't enough enough data for 5-fold cross-validation for training (cold start), just use random data.
            x = np.random.randn(*np.concatenate([data['test_tasks_test_features'][task_idx] for task_idx in range(len(data['test_tasks_test_features']))]).shape)
            y = np.random.uniform(0, 1, len(x))
            corr = np.random.randint(0, 2, len(x))

        kf = ShuffleSplit(n_splits=1, test_size=0.3)
        kf.get_n_splits(x)
        preprocessing = PreProcess(threshold_scaling=True, standard_scaling=True, inside_ball_scaling=False, add_bias=True)

        if settings['val_method'][0] == 'MSE' or settings['val_method'][0] == 'MAE' or settings['val_method'][0] == 'NMSE':
            best_performance = np.Inf
        else:
            best_performance = -1
        best_param = None
        for regul_param in settings['regul_param_range']:
            curr_val_performances = []
            for train_index, test_index in kf.split(x):
                x_tr, x_val = x[train_index], x[test_index]
                y_tr, y_val = y[train_index], y[test_index]
                corr_tr, corr_val = corr[train_index], corr[test_index]

                x_tr, y_tr, _ = preprocessing.transform(x_tr, y_tr, corr_tr, fit=True, multiple_tasks=False)
                x_val, y_val, corr_val = preprocessing.transform(x_val, y_val, corr_val, fit=False, multiple_tasks=False)

                model_itl = ITL(regul_param)
                model_itl.fit(x_tr, y_tr)

                val_predictions = model_itl.predict(x_val)
                val_performance = evaluation_methods(y_val, val_predictions, corr_val, settings['val_method'])[0]
                curr_val_performances.append(val_performance)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                average_val_performance = np.nanmean(curr_val_performances)
            if settings['val_method'][0] == 'MSE' or settings['val_method'][0] == 'MAE' or settings['val_method'][0] == 'NMSE':
                if average_val_performance < best_performance:
                    best_performance = average_val_performance
                    best_param = regul_param
            else:
                if average_val_performance > best_performance:
                    best_performance = average_val_performance
                    best_param = regul_param

        # Retrain on full training set
        x, y, _ = preprocessing.transform(x, y, corr, fit=True, multiple_tasks=False)
        x_test, y_test, corr_test = preprocessing.transform(data['test_tasks_test_features'][task_idx], data['test_tasks_test_labels'][task_idx], data['test_tasks_test_corr'][task_idx], fit=False, multiple_tasks=False)

        model_itl = ITL(best_param)
        model_itl.fit(x, y)

        # Testing
        test_predictions = model_itl.predict(x_test)
        all_predictions.append(test_predictions)
        all_weights.append(model_itl.weight_vector)
        all_performances.append(evaluation_methods(y_test, test_predictions, corr_test,  settings['evaluation']))
    test_performance = np.mean(all_performances, 0)
    # print(f'{"Independent":12s} | test performance: {test_performance[0]:12.5f} | {time() - tt:5.2f}sec')

    return test_performance, all_predictions, all_weights


class ITL:
    def __init__(self, regularization_parameter=1e-2):
        self.regularization_parameter = regularization_parameter
        self.weight_vector = None

    def fit(self, features, labels):
        dims = features.shape[1]

        try:
            weight_vector = lstsq(features.T @ features + self.regularization_parameter * np.eye(dims), features.T @ labels)[0]
        except:
            weight_vector = np.zeros(dims)
        self.weight_vector = weight_vector

    def predict(self, features):
        pred = np.matmul(features, self.weight_vector)
        return pred
