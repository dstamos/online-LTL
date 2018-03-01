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
    data = synthetic_data_gen(data_settings)
    training(data, data_settings, training_settings)


def training(data, data_settings, training_settings):
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
        all_val_perf = [None] * len(param1_range)
        best_val_performance = 10 ** 8
        for param1_idx, param1 in enumerate(param1_range):

            ################################################ #####
            # OPTIMISATION
            D = random.randn(n_dims, n_dims)

            W_pred = np.zeros((n_dims, n_tasks))
            X_train, Y_train = [None] * n_tasks, [None] * n_tasks
            for _, task_idx in enumerate(task_range_tr):
                X_train[task_idx] = np.concatenate((data['X_val'][task_idx], data['X_train'][task_idx]))
                Y_train[task_idx] = np.concatenate((data['Y_val'][task_idx], data['Y_train'][task_idx]))
                n_points = len(Y_train[task_idx])

            batch_objective = lambda D: sum([n_points * norm(pinv(X_train[i] @ D @ X_train[i].T + n_points * eye(n_points)) @ Y_train[i])**2 for i in task_range_tr])
            batch_grad = lambda D: batch_grad_func(D, task_range_tr, data)

            Lipschitz = (6 / (np.sqrt(n_points) * n_points**2)) * max([norm(X_train[i], ord=np.inf)**3 for i in task_range_tr])
            # Lipschitz = (2 / n_points) * max([norm(X_train[i], ord=np.inf)**2 for i in task_range_tr])
            step_size = 1 / Lipschitz
            # step_size = 55

            curr_obj = batch_objective(D)

            objectives = []
            n_iter = 10**10
            curr_tol = 10 ** 10
            conv_tol = 10 ** -5
            c_iter = 0


            while (c_iter < n_iter) and (curr_tol > conv_tol):
                prev_D = D
                prev_obj = curr_obj

                D = prev_D - step_size * batch_grad(prev_D)


                curr_obj = batch_objective(D)
                objectives.append(curr_obj)

                curr_tol = abs(curr_obj - prev_obj) / prev_obj

                if (c_iter % 5000 == 0):
                    # plt.plot(objectives, "b")
                    # plt.pause(0.0001)
                    print("iter: %5d | obj: %20.18f | tol: %20.18f" % (c_iter, curr_obj, curr_tol))
                c_iter = c_iter + 1
            # projection on the 1/lambda trace norm ball
            U, s, Vt = svd(D) # eigen
            # U, s = np.linalg.eig(D)
            s = np.sign(s) * [max(np.abs(s[i]) - param1, 0) for i in range(len(s))]
            D = U @ np.diag(s) @ Vt
            # D = U @ np.diag(s) @ U.T

            plt.figure()
            plt.imshow(D)
            plt.pause(0.01)
            # objectives = np.array(objectives)
            # return D, objectives

            #####################################################
            # VALIDATION
            X_train, Y_train = [None] * n_tasks, [None] * n_tasks
            for _, task_idx in enumerate(task_range_val):
                X_train[task_idx] = data['X_train'][task_idx]
                Y_train[task_idx] = data['Y_train'][task_idx]

            Y_val_pred = [None] * n_tasks
            for _, task_idx in enumerate(task_range_val):
                # fixing D and solving for w_t
                n_points = len(Y_train[task_idx])
                # replace pinv with np.linalg.solve or wahtever

                curr_w_pred = (D @ X_train[task_idx].T @ pinv(X_train[task_idx] @ D @ X_train[task_idx].T + n_points * eye(n_points)) @ Y_train[task_idx]).ravel()
                W_pred[:, task_idx] = curr_w_pred


                Y_val_pred[task_idx] = data['X_val'][task_idx] @ curr_w_pred


            val_perf = mean_squared_error(data['Y_val'], Y_val_pred, task_range_val)
            weight_error = weight_vector_perf(Wtrue, W_pred, task_range_val)

            all_val_perf[param1_idx] = val_perf

            if val_perf < best_val_performance:
                best_val_performance = val_perf

                best_param = param1
                best_D = D
                best_val_prf = val_perf
            print('Batch LTL: param1: %12.10f | val MSE: %7.5f | time: %7.5f' % (param1, val_perf, time.time() - t))
            asdf = 1


        #####################################################
        # TEST
        X_train, Y_train = [None] * n_tasks, [None] * n_tasks
        for _, task_idx in enumerate(task_range_test):
            X_train[task_idx] = np.concatenate((data['X_val'][task_idx], data['X_train'][task_idx]))
            Y_train[task_idx] = np.concatenate((data['Y_val'][task_idx], data['Y_train'][task_idx]))

        Y_test_pred = [None] * n_tasks
        for _, task_idx in enumerate(task_range_test):
            # fixing D and solving for w_t
            n_points = len(Y_train[task_idx])
            curr_w_pred = (best_D @ X_train[task_idx].T @ pinv(X_train[task_idx] @ best_D @ X_train[task_idx].T + n_points * eye(n_points)) @ Y_train[task_idx]).ravel()
            W_pred[:, task_idx] = curr_w_pred
            Y_test_pred[task_idx] = data['X_test'][task_idx] @ curr_w_pred

        test_perf = mean_squared_error(data['Y_test'], Y_test_pred, task_range_test)
        weight_error = weight_vector_perf(Wtrue, W_pred, task_range_test)




        print('Batch ILL: val MSE: %7.5f | test MSE: %7.5f' % (val_perf, test_perf))

    k=1

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

    METHOD = 'batch_LTL'
    # METHOD = 'smart_ITL'
    # METHOD = 'dumb_ITL'




    if len(sys.argv) > 1:
        print("if")
        a = sys.argv[2] # etc
    else:
        from fixed_run_settings import *

    main(DATA_SETTINGS, TRAINING_SETTINGS)
    print("done")