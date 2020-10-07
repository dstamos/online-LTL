import numpy as np
from numpy.linalg import pinv
from numpy import identity as eye
# from src.utilities import mean_squared_error
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
