import numpy as np
from numpy.linalg.linalg import norm, pinv, matrix_power
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class BiasLTL(BaseEstimator):
    def __init__(self, regularization_parameter=1e-2, step_size_bit=1e+3, keep_all_metaparameters=True):
        self.keep_all_metaparameters = keep_all_metaparameters
        self.regularization_parameter = regularization_parameter
        self.step_size_bit = step_size_bit
        self.all_metaparameters_ = None
        self.metaparameter_ = None

    def fit(self, all_features, all_labels, extra_inputs=None):
        extra_inputs = self._check_extra_inputs(extra_inputs)

        all_features, all_labels = check_X_y(all_features, all_labels)

        mean_vector = np.random.randn(all_features.shape[1]) / norm(np.random.randn(all_features.shape[1]))
        all_features, all_labels = self._split_tasks(all_features, extra_inputs['point_indexes_per_task'], all_labels)

        all_metaparameters = [None] * len(all_features)
        for task_idx in range(len(all_features)):
            mean_vector = self.solve_wrt_metaparameter(mean_vector, all_features[task_idx], all_labels[task_idx], curr_iteration=task_idx, inner_iter_cap=3)
            all_metaparameters[task_idx] = mean_vector
        self.all_metaparameters_ = all_metaparameters
        self.metaparameter_ = mean_vector

    def fit_inner(self, all_features, all_labels=None, extra_inputs=None):
        extra_inputs = self._check_extra_inputs(extra_inputs)

        check_is_fitted(self)
        if all_labels is None:
            all_features = check_array(all_features)
            all_features = self._split_tasks(all_features, extra_inputs['re_train_indexes'])
        else:
            all_features, all_labels = check_X_y(all_features, all_labels)
            all_features, all_labels = self._split_tasks(all_features, extra_inputs['re_train_indexes'], all_labels)
        if extra_inputs['predictions_for_each_training_task'] is False:
            weight_vectors = self.metaparameter_
            for task_idx in range(len(all_features)):
                if all_labels is not None:
                    weight_vectors = self.solve_wrt_w(weight_vectors, all_features[task_idx], all_labels[task_idx])  # I don't know if this makes sense, check with Dimitri. The next loop is the same
            return weight_vectors
        else:
            weight_vectors_per_metaparameter = []
            for metaparam_idx in range(len(self.all_metaparameters_)):
                weight_vectors = self.all_metaparameters_[task_idx]
                for task_idx in range(len(all_features)):
                    if all_labels is not None:
                        weight_vectors = self.solve_wrt_w(weight_vectors, all_features[task_idx], all_labels[task_idx])
                weight_vectors_per_metaparameter.append(weight_vectors)
            return weight_vectors_per_metaparameter

    def predict(self, all_features,  extra_inputs=None):
        weight_vectors = self.fit_inner(extra_inputs['re_train_features'], extra_inputs['re_train_labels'], extra_inputs)
        extra_inputs = self._check_extra_inputs(extra_inputs)

        all_features = check_array(all_features)
        all_features = self._split_tasks(all_features, extra_inputs['point_indexes_per_task'])

        if extra_inputs['predictions_for_each_training_task'] is False:
            all_predictions = []
            for task_idx in range(len(all_features)):
                pred = np.matmul(all_features[task_idx], weight_vectors)
                all_predictions.append(pred)
            return all_predictions
        else:
            if self.all_metaparameters_ is None:
                raise ValueError('Not all metaparameters were saved. Refit with predictions_for_each_training_task set to True.')
            all_predictions = []
            for metamodel_idx in range(len(weight_vectors)):
                metamodel_predictions = []
                weights = weight_vectors[i]
                for task_idx in range(len(all_features)):
                    pred = np.matmul(all_features[task_idx], weights)
                    metamodel_predictions.append(pred)
                all_predictions.append(metamodel_predictions)
            return all_predictions

    def solve_wrt_metaparameter(self, h, x, y, curr_iteration=0, inner_iter_cap=10):
        n = len(y)

        def grad(curr_h):
            return 2 * self.regularization_parameter ** 2 * n * x.T @ matrix_power(pinv(x @ x.T + self.regularization_parameter * n * np.eye(n)), 2) @ ((x @ curr_h).ravel() - y)

        i = 0
        curr_iteration = curr_iteration * inner_iter_cap
        while i < inner_iter_cap:
            i = i + 1
            prev_h = h
            curr_iteration = curr_iteration + 1
            step_size = np.sqrt(2) * self.step_size_bit / ((self.step_size_bit + 1) * np.sqrt(curr_iteration))
            h = prev_h - step_size * grad(prev_h)
        return h

    def solve_wrt_w(self, h, x, y):
        n = len(y)
        dims = x.shape[1]
        c_n_lambda = x.T @ x / n + self.regularization_parameter * np.eye(dims)
        w = pinv(c_n_lambda) @ (x.T @ y / n + self.regularization_parameter * h).ravel()

        return w

    @staticmethod
    def _split_tasks(all_features, indexes, all_labels=None):
        # Split the blob/array of features into a list of tasks based on point_indexes_per_task
        all_features = [all_features[indexes == task_idx] for task_idx in np.unique(indexes)]
        if all_labels is None:
            return all_features
        all_labels = [all_labels[indexes == task_idx] for task_idx in np.unique(indexes)]
        return all_features, all_labels

    @staticmethod
    def _check_extra_inputs(extra_inputs):
        if extra_inputs is None:
            extra_inputs = {'predictions_for_each_training_task': False, 'point_indexes_per_task': None}
        if extra_inputs['point_indexes_per_task'] is None:
            raise ValueError("The vector point_indexes_per_task of task idendifiers is necessary.")
        return extra_inputs
