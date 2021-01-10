import numpy as np
from scipy.linalg import lstsq
from numpy.linalg.linalg import norm, pinv, matrix_power


class BiasLTL:
    def __init__(self, regul_param=1e-2, step_size_bit=1e+3, keep_all_metaparameters=True):
        self.keep_all_metaparameters = keep_all_metaparameters
        self.regularization_parameter = regul_param
        self.step_size_bit = step_size_bit
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
