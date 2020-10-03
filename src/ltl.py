import numpy as np
from numpy.linalg.linalg import norm, pinv, matrix_power
from sklearn.metrics import mean_squared_error


def variance_online_ltl(data, training_settings):
    dims = data.dims
    best_mean_vector = np.random.randn(dims) / norm(np.random.randn(dims))

    best_val_performance = np.Inf

    validation_curve = []
    for regularization_parameter in training_settings['regularization_parameter_range']:
        validation_performances = []
        test_performances = []
        all_h = []

        mean_vector = best_mean_vector

        for task_idx in range(len(data.training_tasks)):
            #####################################################
            # Optimisation
            x_train = data.training_tasks[task_idx].training.features
            y_train = data.training_tasks[task_idx].training.labels

            mean_vector = solve_wrt_h(mean_vector, x_train, y_train, regularization_parameter, training_settings['step_size'], curr_iteration=task_idx)
            all_h.append(mean_vector)
            # average_h = np.mean(all_h, axis=0)
            average_h = mean_vector
            #####################################################
            # Test
            # Measure the test error after every training task for the shake of pretty plots at the end
            current_test_errors = []
            for test_task_idx in range(len(data.test_tasks)):
                x_train = data.test_tasks[test_task_idx].training.features
                y_train = data.test_tasks[test_task_idx].training.labels
                w = solve_wrt_w(average_h, x_train, y_train, regularization_parameter)

                x_test = data.test_tasks[test_task_idx].test.features
                y_test = data.test_tasks[test_task_idx].test.labels

                test_perf = mean_squared_error(y_test, x_test @ w)
                current_test_errors.append(test_perf)
            test_performances.append(np.mean(current_test_errors))

        # #####################################################
        # # Validation
        # Validation only needs to be measured at the very end, after we've trained on all training tasks
        for validation_task_idx in range(len(data.validation_tasks)):
            x_train = data.validation_tasks[validation_task_idx].training.features
            y_train = data.validation_tasks[validation_task_idx].training.labels
            w = solve_wrt_w(average_h, x_train, y_train, regularization_parameter)

            x_test = data.validation_tasks[validation_task_idx].test.features
            y_test = data.validation_tasks[validation_task_idx].test.labels

            validation_perf = mean_squared_error(y_test, x_test @ w)
            validation_performances.append(validation_perf)
        validation_performance = np.mean(validation_performances)
        print(f'lambda: {regularization_parameter:6e} | val MSE: {validation_performance:8.5f} | test MSE: {test_perf:8.5f} | norm h: {norm(best_mean_vector):4.2f}')

        validation_curve.append(validation_performance)

        # best_h = np.average(all_h, axis=0)

        if validation_performance < best_val_performance:
            validation_criterion = True
        else:
            validation_criterion = False

        if validation_criterion:
            best_val_performance = validation_performance

            best_mean_vector = average_h
            best_test_performances = test_performances

    print(best_mean_vector)
    results = {'best_mean_vector': best_mean_vector,
               'best_test_performances': best_test_performances}
    return results


def solve_wrt_h(prev_h, x, y, param, step_size_bit, curr_iteration=0):
    n = len(y)

    def grad(curr_h):
        return 2 * param**2 * n * x.T @ matrix_power(pinv(x @ x.T + param * n * np.eye(n)), 2) @ ((x @ curr_h).ravel() - y)

    curr_iteration = curr_iteration + 1
    step_size = np.sqrt(2) * step_size_bit / ((step_size_bit + 1) * np.sqrt(curr_iteration))
    h = prev_h - step_size * grad(prev_h)

    return h


def solve_wrt_w(h, x, y, param):
    n = len(y)
    dims = x.shape[1]

    c_n_lambda = x.T @ x / n + param * np.eye(dims)
    w = c_n_lambda @ (x.T @ y / n + param * h).ravel()

    return w

# def solve_wrt_h_stochastic(h, x, y, param1, step_size_bit):
#     n = len(y)
#
#     obj_fixed_part = pinv(x @ x.T + param1 * n * np.eye(n))
#     grad_fixed_part = 2 * param1**2 * n * x.T @ matrix_power(pinv(x @ x.T + param1 * n * np.eye(n)), 2)
#
#     def obj(curr_h):
#         objective = param1**2 * n * norm(obj_fixed_part @ (x @ curr_h - y)) ** 2
#         return objective
#
#     def grad(curr_h):
#         gradient = grad_fixed_part @ ((x @ curr_h).ravel() - y)
#         return gradient
#
#     all_h = []
#     objectives = []
#
#     curr_iter = 0
#     max_epochs = 1
#     curr_epoch = 0
#     while curr_epoch < max_epochs:
#         curr_epoch = curr_epoch
#         prev_h = h
#
#         curr_iter = curr_iter + 1
#         step_size = step_size_bit / (2*np.sqrt(2)*(step_size_bit + 1)*np.sqrt(curr_iter))
#         h = prev_h - step_size * grad(prev_h)
#         all_h.append(h)
#
#         curr_obj = obj(h)
#         objectives.append(curr_obj)
#
#         print("iter: %5d | obj: %20.15f" % (curr_iter, curr_obj))
#
#     return np.mean(all_h)


def variance_batch_ltl(data, training_settings):
    dims = data.dims
    best_mean_vector = np.random.randn(dims) / norm(np.random.randn(dims))

    best_val_performance = np.Inf

    validation_curve = []
    for regularization_parameter in training_settings['regularization_parameter_range']:
        validation_performances = []
        test_performances = []

        #####################################################
        # Optimisation
        a_matrix = np.zeros((dims, dims))
        b = np.zeros(dims)
        n_training_tasks = len(data.training_tasks)
        for task_idx in range(n_training_tasks):
            x_train = data.training_tasks[task_idx].training.features
            y_train = data.training_tasks[task_idx].training.labels
            n = len(y_train)

            g_matrix = pinv(x_train @ x_train.T + n * regularization_parameter * np.eye(n))

            a_matrix = a_matrix + x_train.T @ g_matrix @ g_matrix @ x_train
            b = b + x_train.T @ g_matrix @ y_train
        a_matrix = 1 / n_training_tasks * a_matrix
        b = 1 / n_training_tasks * b
        mean_vector = pinv(a_matrix) @ b

        #####################################################
        # Validation
        # Validation only needs to be measured at the very end, after we've trained on all training tasks
        for validation_task_idx in range(len(data.validation_tasks)):
            x_train = data.validation_tasks[validation_task_idx].training.features
            y_train = data.validation_tasks[validation_task_idx].training.labels
            w = solve_wrt_w(mean_vector, x_train, y_train, regularization_parameter)

            x_test = data.validation_tasks[validation_task_idx].test.features
            y_test = data.validation_tasks[validation_task_idx].test.labels

            validation_perf = mean_squared_error(y_test, x_test @ w)
            validation_performances.append(validation_perf)
        validation_performance = np.mean(validation_performances)

        #####################################################
        #####################################################
        # Test
        # Measure the test error after every training task for the shake of pretty plots at the end
        current_test_errors = []
        for test_task_idx in range(len(data.test_tasks)):
            x_train = data.test_tasks[test_task_idx].training.features
            y_train = data.test_tasks[test_task_idx].training.labels
            w = solve_wrt_w(mean_vector, x_train, y_train, regularization_parameter)

            x_test = data.test_tasks[test_task_idx].test.features
            y_test = data.test_tasks[test_task_idx].test.labels

            test_perf = mean_squared_error(y_test, x_test @ w)
            current_test_errors.append(test_perf)
        test_performances.append(np.mean(current_test_errors))

        validation_curve.append(validation_performance)

        print(f'lambda: {regularization_parameter:6e} | val MSE: {validation_performance:8.5f} | test MSE: {test_perf:8.5f} | norm h: {norm(best_mean_vector):4.2f}')
        # best_h = np.average(all_h, axis=0)

        if validation_performance < best_val_performance:
            validation_criterion = True
        else:
            validation_criterion = False

        if validation_criterion:
            best_val_performance = validation_performance

            best_mean_vector = mean_vector
            best_test_performances = test_performances

    print(best_mean_vector)
    results = {'best_mean_vector': best_mean_vector,
               'best_test_performances': best_test_performances}
    return results

