from scipy.linalg import lstsq
import numpy as np
from numpy.linalg.linalg import norm
from src.utilities import multiple_tasks_mae_clip
from time import time
from src.preprocessing import PreProcess


def train_test_meta(data, settings, verbose=True):
    # Preprocess the data
    preprocessing = PreProcess(threshold_scaling=True, standard_scaling=True, inside_ball_scaling=True, add_bias=True)
    tr_tasks_tr_features, tr_tasks_tr_labels = preprocessing.transform(data['tr_tasks_tr_features'], data['tr_tasks_tr_labels'], fit=True)

    # Training
    tt = time()
    best_model_ltl = None
    best_param = None
    best_performance = np.Inf
    model_ltl = BiasLTL(keep_all_metaparameters=True)
    for regul_param in settings['regul_param_range']:
        # Optimise metaparameters on the training tasks.
        model_ltl.regularization_parameter = regul_param
        model_ltl.fit_meta(tr_tasks_tr_features, tr_tasks_tr_labels)

        # Check performance on the validation tasks.
        val_tasks_tr_features, val_tasks_tr_labels = preprocessing.transform(data['val_tasks_tr_features'], data['val_tasks_tr_labels'], fit=True)
        val_tasks_val_features, val_tasks_val_labels = preprocessing.transform(data['val_tasks_val_features'], data['val_tasks_val_labels'], fit=False)
        if settings['fine_tune'] is True:
            all_weight_vectors = model_ltl.fine_tune(val_tasks_tr_features, val_tasks_tr_labels, regul_param=regul_param)
            val_task_predictions = model_ltl.predict(val_tasks_val_features, all_weight_vectors)
        else:
            val_task_predictions = model_ltl.predict(val_tasks_val_features)
        val_performance = multiple_tasks_mae_clip(val_tasks_val_labels, val_task_predictions, error_progression=False)
        if val_performance < best_performance:
            best_param = regul_param
            best_performance = val_performance
            best_model_ltl = model_ltl
        if verbose is True:
            print(f'{"LTL":10s} | param: {regul_param:6e} | val performance: {val_performance:12.5f} | {time() - tt:5.2f}sec')

    # Test
    # x_merged = [np.concatenate([data['test_tasks_tr_features'][task_idx], data['test_tasks_val_features'][task_idx]]) for task_idx in range(len(data['test_tasks_indexes']))]
    # y_merged = [np.concatenate([data['test_tasks_tr_labels'][task_idx], data['test_tasks_val_labels'][task_idx]]) for task_idx in range(len(data['test_tasks_indexes']))]
    # test_tasks_tr_features, test_tasks_tr_labels = preprocessing.transform(x_merged, y_merged, fit=True)

    test_tasks_tr_features, test_tasks_tr_labels = preprocessing.transform(data['test_tasks_tr_features'], data['test_tasks_tr_labels'], fit=True)
    test_tasks_val_features, test_tasks_val_labels = preprocessing.transform(data['test_tasks_val_features'], data['test_tasks_val_labels'], fit=False)
    test_tasks_test_features, test_tasks_test_labels = preprocessing.transform(data['test_tasks_test_features'], data['test_tasks_test_labels'], fit=False)
    if settings['fine_tune'] is True:
        all_weight_vectors = best_model_ltl.fine_tune(test_tasks_tr_features, test_tasks_tr_labels, best_param)
        test_task_predictions = best_model_ltl.predict(test_tasks_test_features, all_weight_vectors)

        # all_weight_vectors = [[None] * len(data['test_tasks_indexes'])] * len(best_model_ltl.all_metaparameters_)
        # for task_idx in range(len(data['test_tasks_indexes'])):
        #     best_performance = np.Inf
        #     best_param = None
        #     for regul_param in settings['regul_param_range']:
        #         weight_vector = best_model_ltl.fine_tune([test_tasks_tr_features[task_idx]], [test_tasks_tr_labels[task_idx]], regul_param=regul_param)
        #         val_predictions = best_model_ltl.predict([test_tasks_val_features[task_idx]], weight_vector)
        #         val_performance = multiple_tasks_mae_clip([test_tasks_val_labels[task_idx]], val_predictions, error_progression=False)
        #         if val_performance < best_performance:
        #             best_performance = val_performance
        #             best_param = regul_param
        #     x_merged = np.concatenate([test_tasks_tr_features[task_idx], test_tasks_val_features[task_idx]])
        #     y_merged = np.concatenate([test_tasks_tr_labels[task_idx], test_tasks_val_labels[task_idx]])
        #     weight_vector = best_model_ltl.fine_tune([x_merged], [y_merged], regul_param=best_param)
        #     for meta_param_idx in range(len(best_model_ltl.all_metaparameters_)):
        #         all_weight_vectors[meta_param_idx][task_idx] = weight_vector[meta_param_idx][0]
        # test_task_predictions = best_model_ltl.predict(test_tasks_test_features, all_weight_vectors)
    else:
        test_task_predictions = best_model_ltl.predict(test_tasks_test_features)
    test_performance = multiple_tasks_mae_clip(test_tasks_test_labels, test_task_predictions, error_progression=True)
    print(f'{"LTL":12s} | test performance: {test_performance[-1]:12.5f} | {time() - tt:5.2f}sec')
    return best_model_ltl, test_performance


class BiasLTL:
    def __init__(self, regul_param=1e-2, keep_all_metaparameters=True):
        self.keep_all_metaparameters = keep_all_metaparameters
        self.regularization_parameter = regul_param
        self.all_metaparameters_ = None
        self.metaparameter_ = None

    def fit_meta(self, all_features, all_labels):
        """
        This optimises the metalearning algorithm.
        It recovers the very last metaparameter and all of them in a list (for the purpose of checking how the performance progresses with more training tasks).
        :param all_features: List of features for each task. List length is T. Each component is a (n, d) array.
        :param all_labels: List of labels for each task. List length is T. Each component is a (n, ) array.
        :return:
        """
        if self.metaparameter_ is not None:
            # Initializaton from a previous optimation.
            mean_vector = self.metaparameter_
        else:
            mean_vector = np.random.randn(all_features[0].shape[1]) / norm(np.random.randn(all_features[0].shape[1]))

        all_metaparameters = []
        for task_idx in range(len(all_features)):
            mean_vector = self.solve_wrt_metaparameter(mean_vector, all_features[task_idx], all_labels[task_idx], curr_iteration=task_idx, inner_iter_cap=3)
            if all_metaparameters:
                mean_vector = (task_idx * all_metaparameters[-1] + mean_vector) / (task_idx + 1)
            all_metaparameters.append(mean_vector)

            # if all_metaparameters:
            #     all_metaparameters.append((task_idx * all_metaparameters[-1] + mean_vector) / (task_idx + 1))
            # else:
            #     all_metaparameters.append(mean_vector)
        self.all_metaparameters_ = all_metaparameters
        self.metaparameter_ = mean_vector

    def fine_tune(self, all_features, all_labels, regul_param=None):
        """
        This takes all metaparameters that have been fit already (one for each training task) and recovers fine-tunes T weight vectors.
        Where T is the length of all_features.
        :param regul_param: The regularization parameter
        :param all_features: List of features for each task. List length is T. Each component is a (n, d) array.
        :param all_labels: List of labels for each task. List length is T. Each component is a (n, ) array.
        :return: A list of lists. [[w_task_1, w_tasks_2, w_tasks_3], ..., [w_task_1, w_tasks_2, w_tasks_3]] and this has length len(self.all_metaparameters_
        """
        if regul_param is None:
            regul_param = self.regularization_parameter

        all_weight_vectors = []
        for metaparameter in self.all_metaparameters_:
            current_weight_vectors = []
            for task_idx in range(len(all_features)):
                weight_vector = self.solve_wrt_w(metaparameter, all_features[task_idx], all_labels[task_idx], regul_param)
                current_weight_vectors.append(weight_vector)
            all_weight_vectors.append(current_weight_vectors)
        return all_weight_vectors

    @staticmethod
    def predict(all_features, weight_vectors=None):
        # if weight_vectors is None:
        all_predictions = []
        for all_current_vectors in weight_vectors:
            curr_predictions = []
            for task_idx in range(len(all_features)):
                pred = np.matmul(all_features[task_idx], all_current_vectors[task_idx])
                curr_predictions.append(pred)
            all_predictions.append(curr_predictions)
        return all_predictions

    def solve_wrt_metaparameter(self, h, x, y, curr_iteration=0, inner_iter_cap=10):
        step_size_bit = 1e+3
        n = len(y)
        c_n_hat = x.T @ x / n + self.regularization_parameter * np.eye(x.shape[1])
        x_n_hat = (self.regularization_parameter / np.sqrt(n) * lstsq(c_n_hat.T, x.T)[0]).T
        y_n_hat = 1 / np.sqrt(n) * (y - x @ lstsq(c_n_hat, x.T @ y)[0] / n)

        def grad(curr_h):
            grad_h = x_n_hat.T @ (x_n_hat @ curr_h - y_n_hat)
            return grad_h

        i = 0
        curr_iteration = curr_iteration * inner_iter_cap
        while i < inner_iter_cap:
            i = i + 1
            prev_h = h
            curr_iteration = curr_iteration + 1
            step_size = np.sqrt(2) * step_size_bit / ((step_size_bit + 1) * np.sqrt(curr_iteration))
            h = prev_h - step_size * grad(prev_h)
        return h

    @staticmethod
    def solve_wrt_w(h, x, y, regul_param):
        n = len(y)
        dims = x.shape[1]

        w = lstsq(x.T @ x / n + regul_param * np.eye(dims), (x.T @ y / n + regul_param * h))[0]
        return w
