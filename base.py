import numpy as np

import sys
from utilities import *

from numpy.linalg import pinv
from numpy import identity as eye
from numpy import dot
from numpy.linalg import svd
import time

from sklearn.linear_model import Ridge as ridge
from scipy.optimize import fmin_cg

def main(data_settings, training_settings):
    np.random.seed(data_settings['seed'])

    if data_settings['dataset'] == 'synthetic_regression':
        data = synthetic_data_gen(data_settings)
    training(data, data_settings, training_settings)


def training(data, data_settings, training_settings):
    seed = data_settings['seed']
    method = training_settings['method']
    param1_range = training_settings['param1_range']
    n_tasks = data_settings['n_tasks']
    n_dims = data_settings['n_dims']
    Wtrue = data['W_true']
    task_range = data_settings['task_range']
    task_range_tr = data_settings['task_range_tr']
    task_range_val = data_settings['task_range_val']
    task_range_test = data_settings['task_range_test']
    foldername = training_settings['foldername']
    filename = training_settings['filename']

    t = time.time()
    if method =='dumb_ITL':

        all_train_perf, all_val_perf, all_test_perf = [None] * len(param1_range), [None] * len(param1_range), [None] * len(param1_range)
        best_val_performance = 10**8
        for param1_idx, param1 in enumerate(param1_range):
            W_temp = np.zeros((n_dims, n_tasks))
            Y_train_pred, Y_val_pred, Y_test_pred  = [None] * n_tasks, [None] * n_tasks, [None] * n_tasks
            for _, task_idx in enumerate(task_range_test):

                X_train = data['X_train'][task_idx]
                Y_train = data['Y_train'][task_idx]
                X_val = data['X_val'][task_idx]
                X_test = data['X_test'][task_idx]
                curr_w_pred = (pinv(X_train.T @ X_train + param1 * eye(n_dims)) @ X_train.T @ Y_train).ravel()
                # rr = ridge(fit_intercept=False, alpha=param1)
                # rr.fit(X_train, Y_train)

                Y_train_pred[task_idx] = X_train @ curr_w_pred
                Y_val_pred[task_idx] = X_val @ curr_w_pred
                Y_test_pred[task_idx] = X_test @ curr_w_pred
                W_temp[:, task_idx] = curr_w_pred

            train_perf = mean_squared_error(data['Y_train'], Y_train_pred, task_range_test)
            val_perf = mean_squared_error(data['Y_val'], Y_val_pred, task_range_test)
            test_perf = mean_squared_error(data['Y_test'], Y_test_pred, task_range_test)
            weight_error = weight_vector_perf(Wtrue, W_temp, task_range_test)

            all_train_perf[param1_idx] = train_perf
            all_val_perf[param1_idx] = val_perf
            all_test_perf[param1_idx] = test_perf

            if val_perf < best_val_performance:
                best_val_performance = val_perf

                best_param =param1
                best_W = W_temp
                best_val_prf = val_perf
                best_test_prf = test_perf
        print('Dumb ITL: mean val MSE: %7.5f | mean test MSE: %7.5f' %
              (best_val_prf, best_test_prf))

    elif method == 'smart_ITL':

        param1_range = training_settings['param1_range']
        n_tasks = data_settings['n_tasks']
        n_dims = data_settings['n_dims']
        Wtrue = data['W_true']
        task_range = data_settings['task_range']
        task_range_tr = data_settings['task_range_tr']
        task_range_val = data_settings['task_range_val']
        task_range_test = data_settings['task_range_test']

        all_task_train_perf, all_task_val_perf, all_task_test_perf = [np.nan] * n_tasks, [np.nan] * n_tasks, [np.nan] * n_tasks
        all_train_perf, all_val_perf, all_test_perf = [None] * len(param1_range), [None] * len(param1_range), [None] * len(param1_range)
        for _, task_idx in enumerate(task_range_test):
            best_val_performance = 10 ** 8

            W_temp = np.zeros((n_dims, n_tasks))
            Y_train_pred, Y_val_pred, Y_test_pred = [None] * n_tasks, [None] * n_tasks, [None] * n_tasks
            for param1_idx, param1 in enumerate(param1_range):
                X_train = data['X_train'][task_idx]
                Y_train = data['Y_train'][task_idx]
                X_val = data['X_val'][task_idx]
                X_test = data['X_test'][task_idx]
                curr_w_pred = (pinv(X_train.T @ X_train + param1 * eye(n_dims)) @ X_train.T @ Y_train).ravel()
                # rr = ridge(fit_intercept=False, alpha=param1)
                # rr.fit(X_train, Y_train)

                Y_train_pred[task_idx] = X_train @ curr_w_pred
                Y_val_pred[task_idx] = X_val @ curr_w_pred
                Y_test_pred[task_idx] = X_test @ curr_w_pred
                W_temp[:, task_idx] = curr_w_pred

                train_perf = mean_squared_error_ITL(data['Y_train'], Y_train_pred, [task_idx])
                val_perf = mean_squared_error_ITL(data['Y_val'], Y_val_pred, [task_idx])
                test_perf = mean_squared_error_ITL(data['Y_test'], Y_test_pred, [task_idx])
                # weight_error = weight_vector_perf(Wtrue, W_temp, [task_idx])

                all_train_perf[param1_idx] = train_perf
                all_val_perf[param1_idx] = val_perf
                all_test_perf[param1_idx] = test_perf

                if val_perf < best_val_performance:
                    best_val_performance = val_perf

                    all_task_train_perf[task_idx] = param1

                    all_task_val_perf[task_idx] = val_perf
                    all_task_test_perf[task_idx] = test_perf

        print('Smart ITL: mean val MSE: %7.5f | mean test MSE: %7.5f' %
              (np.nanmean(all_task_val_perf), np.nanmean(all_task_test_perf)))

    elif method == 'batch_LTL':

        param1_range = training_settings['param1_range']
        n_tasks = data_settings['n_tasks']
        n_dims = data_settings['n_dims']
        Wtrue = data['W_true']
        task_range = data_settings['task_range']
        task_range_tr = data_settings['task_range_tr']
        task_range_val = data_settings['task_range_val']
        task_range_test = data_settings['task_range_test']

        # training
        all_train_perf, all_val_perf, all_test_perf = [None] * len(param1_range), [None] * len(param1_range), [None] * len(param1_range)
        best_val_performance = 10 ** 8
        best_D = random.randn(n_dims, n_dims)
        for param1_idx, param1 in enumerate(param1_range):
            W_pred = np.zeros((n_dims, n_tasks))

            #####################################################
            # OPTIMISATION
            X_train, Y_train = [None] * n_tasks, [None] * n_tasks
            for _, task_idx in enumerate(task_range_tr):
                X_train[task_idx] = np.concatenate((data['X_val'][task_idx], data['X_train'][task_idx]))
                Y_train[task_idx] = np.concatenate((data['Y_val'][task_idx], data['Y_train'][task_idx]))
                n_points = len(Y_train[task_idx])

            D = best_D
            D = solve_wrt_D(D, data, X_train, Y_train, n_points, task_range_tr, param1)

            train_perf = mean_squared_error(data['X_val'], data['Y_val'], W_pred, task_range_tr)
            all_train_perf[param1_idx] = train_perf
            #####################################################
            # VALIDATION
            X_train, Y_train = [None] * n_tasks, [None] * n_tasks
            for _, task_idx in enumerate(task_range_val):
                X_train[task_idx] = data['X_train'][task_idx]
                Y_train[task_idx] = data['Y_train'][task_idx]

            W_pred = solve_wrt_w(D, X_train, Y_train, n_tasks, data, W_pred, task_range_val)

            val_perf = mean_squared_error(data['X_val'], data['Y_val'], W_pred, task_range_val)
            all_val_perf[param1_idx] = val_perf

            print('Batch LTL: param1: %8e | val MSE: %7.5f | time: %7.5f' % (param1, val_perf, time.time() - t))

            #####################################################
            # TEST
            X_train, Y_train = [None] * n_tasks, [None] * n_tasks
            for _, task_idx in enumerate(task_range_test):
                X_train[task_idx] = np.concatenate((data['X_val'][task_idx], data['X_train'][task_idx]))
                Y_train[task_idx] = np.concatenate((data['Y_val'][task_idx], data['Y_train'][task_idx]))

            W_pred = solve_wrt_w(D, X_train, Y_train, n_tasks, data, W_pred, task_range_test)

            test_perf = mean_squared_error(data['X_test'], data['Y_test'], W_pred, task_range_test)
            all_test_perf[param1_idx] = test_perf

            if val_perf < best_val_performance:
                best_val_performance = val_perf

                best_param = param1
                best_D = D
                best_val_prf = val_perf
                best_test_prf = test_perf

        results = {}
        results['all_training_errors'] = all_train_perf
        results['all_val_perf'] = all_val_perf
        results['all_test_perf'] = all_test_perf

        save_results(results, data_settings, training_settings, filename, foldername)

        print("best validation stats:")
        print('Batch LTL: param1: %8e | val MSE: %7.5f | test MSE: %7.5f | norm D: %4.2f' % (best_param, best_val_prf, best_test_prf, norm(best_D)))

    elif method == 'online_LTL':

        param1_range = training_settings['param1_range']
        n_tasks = data_settings['n_tasks']
        n_dims = data_settings['n_dims']
        Wtrue = data['W_true']
        task_range = data_settings['task_range']
        task_range_tr = data_settings['task_range_tr']
        task_range_val = data_settings['task_range_val']
        task_range_test = data_settings['task_range_test']

        # training

        T = len(task_range_tr)
        all_train_perf, all_val_perf, all_test_perf = [[] for i in range(T)], [[] for i in range(T)], [[] for i in range(T)]
        best_param, best_train_perf, best_val_perf, best_test_perf = \
            [None] * T, [None] * T, [None] * T, [None] * T
        all_individual_tr_perf = [[] for i in range(T)]

        c_iter = 0
        best_D = random.randn(n_dims, n_dims)
        for pure_task_idx, curr_task_range_tr in enumerate(task_range_tr):

            best_val_performance = 10 ** 8
            for param1_idx, param1 in enumerate(param1_range):
                W_pred = np.zeros((n_dims, n_tasks))

                D = best_D
                #####################################################
                # OPTIMISATION
                X_train, Y_train = [None] * n_tasks, [None] * n_tasks
                for _, task_idx in enumerate([curr_task_range_tr]):
                    X_train[task_idx] = np.concatenate((data['X_val'][task_idx], data['X_train'][task_idx]))
                    Y_train[task_idx] = np.concatenate((data['Y_val'][task_idx], data['Y_train'][task_idx]))
                    n_points = len(Y_train[task_idx])

                D, c_iter = solve_wrt_D_stochastic(D, data, X_train, Y_train, n_points, [curr_task_range_tr], param1, c_iter)

                # Check performance on ALL training tasks for this D
                W_pred = solve_wrt_w(D, data['X_train'], data['Y_train'], n_tasks, data, W_pred, task_range_tr)
                train_perf = mean_squared_error(data['X_test'], data['Y_test'], W_pred, [0])
                all_train_perf[pure_task_idx].append(train_perf)

                individual_tr_perf = [None] * len(task_range_tr)
                for idx, task_idx in enumerate(task_range_tr):
                    individual_tr_perf[idx] = mean_squared_error(data['X_test'], data['Y_test'], W_pred, [task_idx])

                #####################################################
                # VALIDATION
                W_pred = solve_wrt_w(D, data['X_train'], data['Y_train'], n_tasks, data, W_pred, task_range_val)

                val_perf = mean_squared_error(data['X_val'], data['Y_val'], W_pred, task_range_val)
                all_val_perf[pure_task_idx].append(val_perf)

                #####################################################
                # TEST
                X_train, Y_train = [None] * n_tasks, [None] * n_tasks
                for _, task_idx in enumerate(task_range_test):
                    X_train[task_idx] = np.concatenate((data['X_val'][task_idx], data['X_train'][task_idx]))
                    Y_train[task_idx] = np.concatenate((data['Y_val'][task_idx], data['Y_train'][task_idx]))

                W_pred = solve_wrt_w(D, X_train, Y_train, n_tasks, data, W_pred, task_range_test)

                test_perf = mean_squared_error(data['X_test'], data['Y_test'], W_pred, task_range_test)
                all_test_perf[pure_task_idx].append(test_perf)

                if val_perf < best_val_performance:
                    best_val_performance = val_perf

                    best_param[pure_task_idx] = param1
                    best_D = D
                    best_train_perf[pure_task_idx] = train_perf
                    best_val_perf[pure_task_idx] = val_perf
                    best_test_perf[pure_task_idx] = test_perf
                    all_individual_tr_perf[pure_task_idx] = individual_tr_perf

                # print('T: %3d | lambda: %6e | val MSE: %7.5f | test MSE: %7.5f | norm D: %4.2f' %
                #       (curr_task_range_tr, param1, val_perf, test_perf, norm(D)))

            print("best validation stats:")
            print('T: %3d | lambda: %6e | val MSE: %7.5f | test MSE: %7.5f | norm D: %4.2f' %
                  (curr_task_range_tr, best_param[pure_task_idx], best_val_perf[pure_task_idx], best_test_perf[pure_task_idx], norm(best_D)))
            print("")

        results = {}
        results['best_param'] = best_param
        results['best_D'] = best_D
        results['best_train_perf'] = best_train_perf
        results['best_val_perf'] = best_val_perf
        results['best_test_perf'] = best_test_perf
        results['all_individual_tr_perf'] = all_individual_tr_perf

        save_results(results, data_settings, training_settings, filename, foldername)


    # plt.figure()
    # for idx1, _ in enumerate(task_range_tr):
    #     plot_thing = [None] * len(task_range_tr)
    #     for idx2, _ in enumerate(task_range_tr):
    #         plot_thing[idx2] = all_individual_tr_perf[idx2][idx1]
    #     plt.plot(plot_thing)
    # plt.pause(0.5)

    print('done')
    return

    # batch version of VAR regularization
    # training_settings['WpredVARbatch'] = WpredVARbatch
    # outputs = {}
    # outputs['WpredVARbatch'] = WpredVARbatch
    # outputs['objectives'] = objectives




if __name__ == "__main__":

    def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    if len(sys.argv) > 1:
        print("if")
    else:
        seed = 999
        n_points = 1050
        n_tasks = 30
        n_dims = 20
        method_IDX = 1 # 0 batch, 1 online
        dataset_IDX = 0
        if method_IDX == 1:
            c_value = 10**3
        else:
            c_value = np.nan
            lambda_idx = 14

        lambda_range = [10 ** float(i) for i in np.arange(-6, 5, 0.5)]

        data_settings = {}
        data_settings['seed'] = seed
        data_settings['n_points'] = n_points
        data_settings['n_dims'] = n_dims

        data_settings['task_range'] = list(np.arange(0, n_tasks))
        data_settings['n_tasks'] = len(data_settings['task_range'])
        data_settings['task_range_tr'] = list(split(range(n_tasks), 3))[0]
        data_settings['task_range_val'] = list(split(range(n_tasks), 3))[1]
        data_settings['task_range_test'] = list(split(range(n_tasks), 3))[2]

        data_settings['train_perc'] = 0.75
        data_settings['val_perc'] = 0.25
        data_settings['noise'] = 0.25

        training_settings = {}
        if method_IDX == 1:
            training_settings['param1_range'] = lambda_range
        else:
            training_settings['param1_range'] = [lambda_range[lambda_idx]]
        # setting for step size on online LTL
        training_settings['c_value'] = c_value
        # setting for non-online methods
        training_settings['conv_tol'] = 10 ** -5
        if method_IDX == 0:
            training_settings['method'] = 'batch_LTL'
            training_settings['filename'] = "seed_" + str(seed) + '-lambda_' + str(training_settings['param1_range'][0])
        elif method_IDX == 1:
            training_settings['method'] = 'online_LTL'
            training_settings['filename'] = "seed_" + str(seed) + '-c_value_' + str(c_value)
        elif method_IDX == 2:
            pass

        if dataset_IDX == 0:
            data_settings['dataset'] = 'synthetic_regression'

            training_settings['foldername'] = 'results/' + data_settings['dataset'] + '-T_' + \
                                              str(n_tasks) + '-n_' + str(n_points) + '/' \
                                              + training_settings['method']
        elif dataset_IDX == 1:
            pass



    # dataset, method, c_value
    # TODO for online LTL remove the lambda_idx thing



    main(data_settings, training_settings)
    print("done")