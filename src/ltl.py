import numpy as np
import time
from numpy.linalg.linalg import norm, pinv, matrix_power
from sklearn.metrics import mean_squared_error


def variance_online_ltl(data, training_settings):
    dims = data.dims
    best_h = np.random.randn(dims) / norm(np.random.randn(dims))

    best_val_performance = np.Inf

    validation_curve = []
    for regularization_parameter in training_settings['regularization_parameter_range']:
        training_performances = []
        validation_performances = []
        test_performances = []
        all_h = []

        h = best_h

        import matplotlib.pyplot as plt

        tt = time.time()
        for task_idx in range(len(data.training_tasks)):
            #####################################################
            # Optimisation
            x_train = data.training_tasks[task_idx].training.features
            y_train = data.training_tasks[task_idx].training.labels

            h = solve_wrt_h_stochastic(h, x_train, y_train, regularization_parameter, training_settings['step_size'], curr_iteration=task_idx)
            all_h.append(h)
            # average_h = np.mean(all_h, axis=0)
            average_h = h
            #####################################################
            # Test
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

        # plt.clf()
        plt.plot(test_performances)
        plt.pause(0.1)

        print('lambda: %6e | test MSE: %7.5f' % (regularization_parameter, np.mean(test_performances)))

        # #####################################################
        # # Validation
        # W_pred = solve_wrt_w(h, data['X_train'], data['Y_train'], n_tasks, data, W_pred, param1, task_range_val)
        #
        # val_perf = mean_squared_error(data_settings, data['X_val'], data['Y_val'], W_pred, W_true, task_range_val)
        # all_val_perf[pure_task_idx].append(val_perf)
        #
        # print('lambda: %6e | val MSE: %7.5f | test MSE: %7.5f | norm D: %4.2f' % (param1, val_perf, test_perf, norm(best_h)))
        #
        # validation_curve[param1_idx] = val_perf
        #
        # best_h = np.average(all_h[:pure_task_idx], axis=0)
        #
        # if val_perf < best_val_performance:
        #     validation_criterion = True
        # else:
        #     validation_criterion = False
        #
        # if validation_criterion:
        #     best_val_performance = val_perf
        #
        #     best_test = test_perf
        #     best_train_perf = all_train_perf
        #     best_val_perf = all_val_perf
        #     best_test_perf = all_test_perf[param1_idx]
    plt.show()

    results = {}
    results['best_val_perf'] = best_val_perf
    results['best_test_perf'] = best_test_perf
    results['time_lapsed'] = time_lapsed


def solve_wrt_h_stochastic(prev_h, x, y, param, step_size_bit, curr_iteration=0):
    n = len(y)
    dims = x.shape[1]

    # grad_fixed_part = 2 * param1**2 * n * x.T @ matrix_power(pinv(x @ x.T + param1 * n * np.eye(n)), 2)

    c_n_lambda = x.T @ x / n + param * np.eye(dims)
    x_hat = (param / np.sqrt(n)) * x @ pinv(c_n_lambda)
    y_hat = (1 / np.sqrt(n)) * (y - x @ pinv(c_n_lambda) @ (x.T @ y) / n)

    def grad(curr_h):
        gradient = x_hat.T @ (x_hat @ curr_h - y_hat)
        return gradient

    # grad_fixed_part = 2 * param1**2 * n * x.T @ matrix_power(pinv(x @ x.T + param1 * n * np.eye(n)), 2)
    #
    # def grad(curr_h):
    #     gradient = grad_fixed_part @ ((x @ curr_h).ravel() - y)
    #     return gradient

    curr_iteration = curr_iteration + 1
    # step_size = 0.5 / curr_iteration
    step_size = step_size_bit / (2*np.sqrt(2)*(step_size_bit + 1)*np.sqrt(curr_iteration))
    granada = grad(prev_h)
    h = prev_h - step_size * granada
    # print(norm(granada))

    return h


def solve_wrt_w(h, x, y, param):
    n = len(y)
    dims = x.shape[1]

    c_n_lambda = x.T @ x / n + param * np.eye(dims)
    # w = (pinv(x.T @ x + param1 * n * np.eye(dims)) @ (n * param1 * h + x.T @ y)).ravel()
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
