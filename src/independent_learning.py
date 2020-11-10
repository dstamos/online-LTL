import numpy as np
from numpy.linalg import pinv
from numpy import identity as eye
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error


def itl(data, training_settings):
    dims = data.dims
    best_weight_vectors = [None] * len(data.test_tasks)
    for task_idx in range(len(data.test_tasks)):
        best_val_performance = np.Inf

        x_train = data.test_tasks[task_idx].training.features
        y_train = data.test_tasks[task_idx].training.labels

        x_val = data.test_tasks[task_idx].validation.features
        y_val = data.test_tasks[task_idx].validation.labels

        for regularization_parameter in training_settings['regularization_parameter_range']:
            #####################################################
            # Optimisation
            curr_w = pinv(x_train.T @ x_train + regularization_parameter * eye(dims)) @ x_train.T @ y_train

            #####################################################
            # Validation
            val_performance = mean_squared_error(y_val, x_val @ curr_w)

            if val_performance < best_val_performance:
                validation_criterion = True
            else:
                validation_criterion = False

            if validation_criterion:
                best_val_performance = val_performance

                best_weight_vectors[task_idx] = curr_w

                best_regularization_parameter = regularization_parameter
                best_val_perf = val_performance
        print('task: %3d | best lambda: %6e | val MSE: %8.5f' % (task_idx, best_regularization_parameter, best_val_perf))
    #####################################################
    # Testing
    test_perfomances = [None] * len(data.test_tasks)
    for task_idx in range(len(data.test_tasks)):
        x_test = data.test_tasks[task_idx].test.features
        y_test = data.test_tasks[task_idx].test.labels
        test_perfomances[task_idx] = mean_squared_error(y_test, x_test @ best_weight_vectors[task_idx])
    print('final test MSE: %8.5f' % (np.mean(test_perfomances)))

    results = {'test_perfomance': np.mean(test_perfomances)}

    return results


class ITL(BaseEstimator):
    def __init__(self, regularization_parameter=1e-2):
        self.regularization_parameter = regularization_parameter
        self.all_parameters_ = None

    def fit(self, all_features, all_labels, extra_inputs=None):
        extra_inputs = self._check_extra_inputs(extra_inputs)

        all_features, all_labels = check_X_y(all_features, all_labels)
        all_features, all_labels = self._split_tasks(all_features, extra_inputs['point_indexes_per_task'], all_labels)

        all_parameters = []
        for task_idx in range(len(all_features)):
            x_train = all_features[task_idx]
            y_train = all_labels[task_idx]
            dims = x_train.shape[1]

            curr_w = pinv(x_train.T @ x_train + self.regularization_parameter * eye(dims)) @ x_train.T @ y_train
            all_parameters.append(curr_w)
        self.all_parameters_ = all_parameters

    def predict(self, all_features, extra_inputs=None):
        extra_inputs = self._check_extra_inputs(extra_inputs)

        all_features = check_array(all_features)
        all_features = self._split_tasks(all_features, extra_inputs['point_indexes_per_task'])

        all_predictions = []
        for task_idx in range(len(all_features)):
            pred = np.matmul(all_features[task_idx], self.all_parameters_[task_idx])
            all_predictions.append(pred)
        return all_predictions

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
            extra_inputs = {'point_indexes_per_task': None}
        if extra_inputs['point_indexes_per_task'] is None:
            raise ValueError("The vector point_indexes_per_task of task idendifiers is necessary.")
        return extra_inputs
