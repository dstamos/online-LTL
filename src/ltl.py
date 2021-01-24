from scipy.linalg import lstsq
import numpy as np
from numpy.linalg.linalg import norm
from src.utilities import multiple_tasks_mae_clip
from time import time
from src.preprocessing import PreProcess
from sklearn.model_selection import KFold


def train_test_meta(data, settings, verbose=True):
    # Create the test tasks here. It's done early to know if we have enough training points on the test tasks for a non-"cold start" problem.
    x_test_tasks_merged = [np.concatenate([data['test_tasks_tr_features'][task_idx], data['test_tasks_val_features'][task_idx]]) for task_idx in range(len(data['test_tasks_indexes']))]
    y_test_tasks_merged = [np.concatenate([data['test_tasks_tr_labels'][task_idx], data['test_tasks_val_labels'][task_idx]]) for task_idx in range(len(data['test_tasks_indexes']))]

    # Preprocess the data
    if settings['fine_tune'] is True and np.all([len(y_test_tasks_merged[task_idx]) > 5 for task_idx in range(len(y_test_tasks_merged))]):
        preprocessing = PreProcess(threshold_scaling=True, standard_scaling=True, inside_ball_scaling=True, add_bias=True)
    else:
        # In the case we don't have training data on the test tasks or we just don't want to fine-tune.
        preprocessing = PreProcess(threshold_scaling=False, standard_scaling=False, inside_ball_scaling=True, add_bias=True)
    tr_tasks_tr_features, tr_tasks_tr_labels = preprocessing.transform(data['tr_tasks_tr_features'], data['tr_tasks_tr_labels'], fit=True, multiple_tasks=True)

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
        x_val_tasks_merged = [np.concatenate([data['val_tasks_tr_features'][task_idx], data['val_tasks_val_features'][task_idx]]) for task_idx in range(len(data['validation_tasks_indexes']))]
        y_val_tasks_merged = [np.concatenate([data['val_tasks_tr_labels'][task_idx], data['val_tasks_val_labels'][task_idx]]) for task_idx in range(len(data['validation_tasks_indexes']))]

        _, _ = preprocessing.transform(x_val_tasks_merged, y_val_tasks_merged, fit=True, multiple_tasks=True)
        val_tasks_test_features, val_tasks_test_labels = preprocessing.transform(data['val_tasks_test_features'], data['val_tasks_test_labels'], fit=False, multiple_tasks=True)
        if settings['fine_tune'] is True and np.all([len(y_test_tasks_merged[task_idx]) > 5 for task_idx in range(len(y_test_tasks_merged))]):
            # If we don't have enough data on the test tasks to fine-tune, the training process should not fine-tune on the validation tasks either.
            all_weight_vectors = model_ltl.fine_tune(x_val_tasks_merged, y_val_tasks_merged, settings['regul_param_range'], preprocessing)
            val_task_predictions = model_ltl.predict(val_tasks_test_features, all_weight_vectors)
        else:
            val_task_predictions = model_ltl.predict(val_tasks_test_features)
        val_performance = multiple_tasks_mae_clip(val_tasks_test_labels, val_task_predictions, error_progression=False)
        if val_performance < best_performance:
            best_param = regul_param
            best_performance = val_performance
            best_model_ltl = model_ltl
        if verbose is True:
            print(f'{"LTL":10s} | param: {regul_param:6e} | val performance: {val_performance:12.5f} | {time() - tt:5.2f}sec')

    # Test
    if settings['fine_tune'] is True and np.all([len(y_test_tasks_merged[task_idx]) > 5 for task_idx in range(len(y_test_tasks_merged))]):
        _, _ = preprocessing.transform(x_test_tasks_merged, y_test_tasks_merged, fit=True, multiple_tasks=True)
    test_tasks_test_features, test_tasks_test_labels = preprocessing.transform(data['test_tasks_test_features'], data['test_tasks_test_labels'], fit=False, multiple_tasks=True)
    if settings['fine_tune'] is True and np.all([len(y_test_tasks_merged[task_idx]) > 5 for task_idx in range(len(y_test_tasks_merged))]):
        all_weight_vectors = best_model_ltl.fine_tune(x_test_tasks_merged, y_test_tasks_merged, settings['regul_param_range'], preprocessing)
        test_task_predictions = best_model_ltl.predict(test_tasks_test_features, all_weight_vectors)
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
        self.all_metaparameters_ = all_metaparameters
        self.metaparameter_ = mean_vector

    def fine_tune(self, all_features, all_labels, regul_param, preprocessing):
        """
        This takes all metaparameters that have been fit already (one for each training task) and recovers fine-tunes T weight vectors.
        Where T is the length of all_features.
        :param preprocessing: The preprocessing class instance.
        :param regul_param: The regularization parameter. If it's a scalar, then there is not cross-validation.
        :param all_features: List of features for each task. List length is T. Each component is a (n, d) array.
        :param all_labels: List of labels for each task. List length is T. Each component is a (n, ) array.
        :return: A list of lists. [[w_task_1, w_tasks_2, w_tasks_3], ..., [w_task_1, w_tasks_2, w_tasks_3]] and this has length len(self.all_metaparameters_
        """
        if np.isscalar(regul_param):
            # Just use the regularization parameter that was given to the function
            best_regul_params = [regul_param] * len(all_features)
        else:
            # In this case for each task, split the given data and do cross validation. The retrain on all and return the corresponding weight vectors.
            best_regul_params = [None] * len(all_features)
            for task_idx in range(len(all_features)):
                kf = KFold(n_splits=5)
                kf.get_n_splits(all_features[task_idx])

                best_performance = np.Inf
                for curr_regul_param in regul_param:
                    curr_val_performances = []
                    for train_index, test_index in kf.split(all_features[task_idx]):
                        x_tr, x_val = all_features[task_idx][train_index], all_features[task_idx][test_index]
                        y_tr, y_val = all_labels[task_idx][train_index], all_labels[task_idx][test_index]

                        x_tr, y_tr = preprocessing.transform(x_tr, y_tr, fit=True, multiple_tasks=False)
                        x_val, y_val = preprocessing.transform(x_val, y_val, fit=False, multiple_tasks=False)

                        weight_vector = self.solve_wrt_w(self.metaparameter_, x_tr, y_tr, curr_regul_param)

                        val_predictions = x_val @ weight_vector
                        val_performance = multiple_tasks_mae_clip([y_val], [[val_predictions]])
                        curr_val_performances.append(val_performance)
                    average_val_performance = np.min(curr_val_performances)
                    if average_val_performance < best_performance:
                        best_performance = average_val_performance
                        best_regul_params[task_idx] = curr_regul_param

        all_features, all_labels = preprocessing.transform(all_features, all_labels, fit=True, multiple_tasks=True)

        all_weight_vectors = []
        for metaparameter in self.all_metaparameters_:
            current_weight_vectors = []
            for task_idx in range(len(all_features)):
                weight_vector = self.solve_wrt_w(metaparameter, all_features[task_idx], all_labels[task_idx], best_regul_params[task_idx])
                current_weight_vectors.append(weight_vector)
            all_weight_vectors.append(current_weight_vectors)

        return all_weight_vectors

    def predict(self, all_features, weight_vectors=None):
        if weight_vectors is None:
            # If weight_vectors is None, then we just use the raw metaparameters as the weight vectors. It's either the case of no fine-tuning or not enough data to fine-tune.
            weight_vectors = [[self.all_metaparameters_[meta_model_idx]] * len(all_features) for meta_model_idx in range(len(self.all_metaparameters_))]

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
