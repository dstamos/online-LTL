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

    if DATASET == 'synthetic_regression':
        data = synthetic_data_gen(data_settings)
    training(data, data_settings, training_settings)


def training(data, data_settings, training_settings):
    seed = data_settings['seed']
    param1_range = training_settings['param1_range']
    n_tasks = data_settings['n_tasks']
    n_dims = data_settings['n_dims']
    Wtrue = data['W_true']
    task_range = data_settings['task_range']
    task_range_tr = data_settings['task_range_tr']
    task_range_val = data_settings['task_range_val']
    task_range_test = data_settings['task_range_test']

    t = time.time()
    if METHOD =='dumb_ITL':

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

    elif METHOD == 'smart_ITL':

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

                train_perf = mean_squared_error(data['Y_train'], Y_train_pred, [task_idx])
                val_perf = mean_squared_error(data['Y_val'], Y_val_pred, [task_idx])
                test_perf = mean_squared_error(data['Y_test'], Y_test_pred, [task_idx])
                weight_error = weight_vector_perf(Wtrue, W_temp, [task_idx])

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

    elif METHOD == 'batch_LTL':

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
        for param1_idx, param1 in enumerate(param1_range):
            W_pred = np.zeros((n_dims, n_tasks))

            #####################################################
            # OPTIMISATION
            X_train, Y_train = [None] * n_tasks, [None] * n_tasks
            for _, task_idx in enumerate(task_range_tr):
                X_train[task_idx] = np.concatenate((data['X_val'][task_idx], data['X_train'][task_idx]))
                Y_train[task_idx] = np.concatenate((data['Y_val'][task_idx], data['Y_train'][task_idx]))
                n_points = len(Y_train[task_idx])

            D = random.randn(n_dims, n_dims)
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

            if val_perf < best_val_performance:
                best_val_performance = val_perf

                best_param = param1
                best_D = D
                best_val_prf = val_perf
            print('Batch LTL: param1: %12.10f | val MSE: %7.5f | time: %7.5f' % (param1, val_perf, time.time() - t))

            #####################################################
            # TEST
            X_train, Y_train = [None] * n_tasks, [None] * n_tasks
            for _, task_idx in enumerate(task_range_test):
                X_train[task_idx] = np.concatenate((data['X_val'][task_idx], data['X_train'][task_idx]))
                Y_train[task_idx] = np.concatenate((data['Y_val'][task_idx], data['Y_train'][task_idx]))

            W_pred = solve_wrt_w(D, X_train, Y_train, n_tasks, data, W_pred, task_range_test)

            test_perf = mean_squared_error(data['X_test'], data['Y_test'], W_pred, task_range_test)
            all_test_perf[param1_idx] = test_perf

        results = {}
        results['all_training_errors'] = all_train_perf
        results['all_val_perf'] = all_val_perf
        results['all_test_perf'] = all_test_perf

        save_results(results, data_settings, training_settings, DATASET)

        print('Batch LTL: val MSE: %7.5f | test MSE: %7.5f' % (val_perf, test_perf))

    elif METHOD == 'online_LTL':

        param1_range = training_settings['param1_range']
        n_tasks = data_settings['n_tasks']
        n_dims = data_settings['n_dims']
        Wtrue = data['W_true']
        task_range = data_settings['task_range']
        task_range_tr = data_settings['task_range_tr']
        task_range_val = data_settings['task_range_val']
        task_range_test = data_settings['task_range_test']

        param1 = param1_range[PARAM1_IDX]

        # training
        all_train_perf, all_val_perf, all_test_perf = [None] * len(task_range_tr), [None] * len(task_range_tr), [None] * len(task_range_tr)

        c_iter = 0
        D = random.randn(n_dims, n_dims)
        for pure_task_idx, curr_task_range_tr in enumerate(task_range_tr):

            W_pred = np.zeros((n_dims, n_tasks))
            #####################################################
            # OPTIMISATION
            X_train, Y_train = [None] * n_tasks, [None] * n_tasks
            for _, task_idx in enumerate([curr_task_range_tr]):
                X_train[task_idx] = np.concatenate((data['X_val'][task_idx], data['X_train'][task_idx]))
                Y_train[task_idx] = np.concatenate((data['Y_val'][task_idx], data['Y_train'][task_idx]))
                n_points = len(Y_train[task_idx])

            # D = solve_wrt_D(D, data, X_train, Y_train, n_points, [curr_task_range_tr], param1)
            D, c_iter = solve_wrt_D_stochastic(D, data, X_train, Y_train, n_points, [curr_task_range_tr], param1, c_iter)

            print(D[0, :5])
            # print(c_iter)
            ############## this is not correct (0 prediction for future tasks)
            # need to take task_range_tr[pure_task_idx:] and predict a w for each of them based on our current D
            train_perf = mean_squared_error(data['X_val'], data['Y_val'], W_pred, task_range_tr)
            ##############
            all_train_perf[pure_task_idx] = train_perf
            #####################################################
            # VALIDATION
            X_train, Y_train = [None] * n_tasks, [None] * n_tasks
            for _, task_idx in enumerate(task_range_val):
                X_train[task_idx] = data['X_train'][task_idx]
                Y_train[task_idx] = data['Y_train'][task_idx]

            W_pred = solve_wrt_w(D, X_train, Y_train, n_tasks, data, W_pred, task_range_val)

            val_perf = mean_squared_error(data['X_val'], data['Y_val'], W_pred, task_range_val)
            all_val_perf[pure_task_idx] = val_perf

            #####################################################
            # TEST
            X_train, Y_train = [None] * n_tasks, [None] * n_tasks
            for _, task_idx in enumerate(task_range_test):
                X_train[task_idx] = np.concatenate((data['X_val'][task_idx], data['X_train'][task_idx]))
                Y_train[task_idx] = np.concatenate((data['Y_val'][task_idx], data['Y_train'][task_idx]))

            W_pred = solve_wrt_w(D, X_train, Y_train, n_tasks, data, W_pred, task_range_test)

            test_perf = mean_squared_error(data['X_test'], data['Y_test'], W_pred, task_range_test)
            all_test_perf[pure_task_idx] = test_perf

            print('online LTL (#T: %3d): val MSE: %7.5f | test MSE: %7.5f' % (curr_task_range_tr, val_perf, test_perf))

        results = {}
        results['all_training_errors'] = all_train_perf
        results['all_val_perf'] = all_val_perf
        results['all_test_perf'] = all_test_perf

        save_results(results, data_settings, training_settings, DATASET)


        # fmin_cg
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin_cg.html



        # print('Smart ITL: mean val MSE: %7.5f | mean test MSE: %7.5f' %
        #       (np.nanmean(all_task_val_perf), np.nanmean(all_task_test_perf)))

    plt.figure()
    plt.plot(all_train_perf, 'r')
    plt.plot(all_val_perf, 'b')
    plt.plot(all_test_perf, 'k')
    plt.pause(0.01)

    k = 1



    # batch version of VAR regularization
    # training_settings['WpredVARbatch'] = WpredVARbatch
    # outputs = {}
    # outputs['WpredVARbatch'] = WpredVARbatch
    # outputs['objectives'] = objectives

    print('done')

    # outputs = []
    # for task_idx, task in enumerate(task_range):
    #     outputs.append(parameter_selection(data, data_settings, training_settings, task_idx))


if __name__ == "__main__":

    # METHOD = 'smart_ITL'
    # METHOD = 'dumb_ITL'
    # METHOD = 'batch_LTL'
    METHOD = 'online_LTL'

    DATASET = 'synthetic_regression'


    if len(sys.argv) > 1:
        print("if")
        PARAM1_IDX = sys.argv[1] # etc
    else:
        from fixed_run_settings import *

        PARAM1_IDX = 4
        # TRAINING_SETTINGS['param1_range'] = [TRAINING_SETTINGS['param1_range'][10]]



    main(DATA_SETTINGS, TRAINING_SETTINGS)
    print("done")