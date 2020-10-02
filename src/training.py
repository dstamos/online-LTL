from src.utilities import *

from numpy.linalg import pinv
import time


def training(data, data_settings, training_settings):
    method = training_settings['method']

    # print("    seed: " + str(seed))
    # print("n_points: " + str(n_points))
    # print(" n_tasks: " + str(n_tasks))
    # print("  n_dims: " + str(n_dims))
    # print("  method: " + str(training_settings['method']))
    # print(" dataset: " + str(data_settings['dataset']))
    # print(" c_value: " + str(c_value))

    if method =='Validation_ITL':
        validation_ITL(data, data_settings, training_settings)

    elif method == 'MTL':
        mtl(data, data_settings, training_settings)

    elif method == 'batch_LTL':
        batch_LTL(data, data_settings, training_settings)

    elif method == 'online_LTL':
        online_LTL(data, data_settings, training_settings)

    print('done')
    return


def validation_ITL(data, data_settings, training_settings):
    foldername = training_settings['foldername']
    filename = training_settings['filename']
    param1_range = training_settings['param1_range']
    n_tasks = data_settings['n_tasks']
    n_dims = data_settings['n_dims']
    task_range = data_settings['task_range']
    task_range_tr = data_settings['task_range_tr']
    task_range_val = data_settings['task_range_val']
    task_range_test = data_settings['task_range_test']

    param1 = param1_range[0]

    T = len(task_range_test)
    all_train_perf, all_val_perf, all_test_perf = [None] * T, [None] * T, [None] * T

    time_lapsed = [None] * T
    # curr_task_range_tr = [] ##################
    curr_task_range_tr = task_range_val
    for pure_task_idx, new_tr_task in enumerate([0.0]):
        # for pure_task_idx, new_tr_task in enumerate(task_range_test): ##################
        t = time.time()
        W_pred = np.zeros((n_dims, n_tasks))
        # curr_task_range_tr.append(new_tr_task) ##################

        #####################################################
        # OPTIMISATION
        X_train, Y_train = [None] * n_tasks, [None] * n_tasks
        n_points = [0] * n_tasks
        for _, task_idx in enumerate(curr_task_range_tr):
            X_train[task_idx] = data['X_train'][task_idx]
            Y_train[task_idx] = data['Y_train'][task_idx]
            n_points[task_idx] = len(Y_train[task_idx])

        D = np.eye(n_dims, n_dims)

        s, U = np.linalg.eigh(D)
        # s = [min(max(si,0),1/param1) for si in s]
        s = [max(si, 0) for si in s]
        s = s / sum(s) * 1 / param1
        D = U @ np.diag(s) @ U.T

        time_lapsed[pure_task_idx] = time.time() - t
        # Check performance on ALL training tasks for this D
        W_pred = solve_wrt_w(D, data['X_train'], data['Y_train'], n_tasks, data, W_pred, curr_task_range_tr)
        train_perf = mean_squared_error(data['X_train'], data['Y_train'], W_pred, curr_task_range_tr)
        all_train_perf[pure_task_idx] = train_perf

        #####################################################
        # VALIDATION
        val_perf = mean_squared_error(data['X_val'], data['Y_val'], W_pred, curr_task_range_tr)
        all_val_perf[pure_task_idx] = val_perf

        #####################################################
        # TEST
        W_pred = solve_wrt_w(D, data['X_train'], data['Y_train'], n_tasks, data, W_pred, task_range_test)
        test_perf = mean_squared_error(data['X_test'], data['Y_test'], W_pred, task_range_test)
        all_test_perf[pure_task_idx] = test_perf

        print("Validation ITL | best validation stats:")
        print('T: %3d | lambda: %6e | val MSE: %7.5f | test MSE: %7.5f | norm D: %4.2f' %
              (new_tr_task, param1, all_val_perf[pure_task_idx], all_test_perf[pure_task_idx], norm(D)))
        print("")

    results = {}
    # results['D'] = D
    results['all_train_perf'] = all_train_perf
    results['all_val_perf'] = all_val_perf
    results['all_test_perf'] = all_test_perf
    results['time_lapsed'] = time_lapsed

    save_results(results, data_settings, training_settings, filename, foldername)

    print("best validation stats:")
    print('Validation ITL: param1: %8e | val MSE: %7.5f | test MSE: %7.5f | norm D: %4.2f' %
          (param1, all_val_perf[pure_task_idx], all_test_perf[pure_task_idx], norm(D)))


def mtl(data, data_settings, training_settings):
    foldername = training_settings['foldername']
    filename = training_settings['filename']
    param1_range = training_settings['param1_range']
    n_tasks = data_settings['n_tasks']
    n_dims = data_settings['n_dims']
    task_range = data_settings['task_range']
    task_range_tr = data_settings['task_range_tr']
    task_range_val = data_settings['task_range_val']
    task_range_test = data_settings['task_range_test']

    param1 = param1_range[0]

    T = len(task_range_test)
    all_train_perf, all_val_perf, all_test_perf = [None] * T, [None] * T, [None] * T

    time_lapsed = [None] * T
    D = random.randn(n_dims, n_dims)
    curr_task_range_tr = task_range_test
    for pure_task_idx, new_tr_task in enumerate([0.0]):
        t = time.time()
        W_pred = np.zeros((n_dims, n_tasks))

        #####################################################
        # OPTIMISATION
        X_train, Y_train = [None] * n_tasks, [None] * n_tasks
        n_points = [0] * n_tasks
        for _, task_idx in enumerate(curr_task_range_tr):
            X_train[task_idx] = data['X_train'][task_idx]
            Y_train[task_idx] = data['Y_train'][task_idx]
            n_points[task_idx] = len(Y_train[task_idx])

        D = solve_wrt_D(D, training_settings, data, X_train, Y_train, n_points, curr_task_range_tr, param1)

        time_lapsed[pure_task_idx] = time.time() - t
        # Check performance on ALL training tasks for this D
        W_pred = solve_wrt_w(D, data['X_train'], data['Y_train'], n_tasks, data, W_pred, task_range_test)
        train_perf = mean_squared_error(data['X_train'], data['Y_train'], W_pred, task_range_test)
        all_train_perf[pure_task_idx] = train_perf

        #####################################################
        # VALIDATION
        val_perf = mean_squared_error(data['X_val'], data['Y_val'], W_pred, task_range_test)
        all_val_perf[pure_task_idx] = val_perf

        #####################################################
        # TEST
        test_perf = mean_squared_error(data['X_test'], data['Y_test'], W_pred, task_range_test)
        all_test_perf[pure_task_idx] = test_perf

        print("MTL | best validation stats:")
        print('T: %3d | lambda: %6e | val MSE: %7.5f | test MSE: %7.5f | norm D: %4.2f' %
              (new_tr_task, param1, all_val_perf[pure_task_idx], all_test_perf[pure_task_idx], norm(D)))
        print("")

    results = {}
    # results['D'] = D
    results['all_train_perf'] = all_train_perf
    results['all_val_perf'] = all_val_perf
    results['all_test_perf'] = all_test_perf
    results['time_lapsed'] = time_lapsed

    save_results(results, data_settings, training_settings, filename, foldername)

    print("best validation stats:")
    print('MTL: param1: %8e | val MSE: %7.5f | test MSE: %7.5f | norm D: %4.2f' %
          (param1, all_val_perf[pure_task_idx], all_test_perf[pure_task_idx], norm(D)))


def batch_LTL(data, data_settings, training_settings):
    foldername = training_settings['foldername']
    filename = training_settings['filename']
    param1_range = training_settings['param1_range']
    n_tasks = data_settings['n_tasks']
    n_dims = data_settings['n_dims']
    task_range = data_settings['task_range']
    task_range_tr = data_settings['task_range_tr']
    task_range_val = data_settings['task_range_val']
    task_range_test = data_settings['task_range_test']

    param1 = param1_range[0]

    T = len(task_range_tr)
    all_train_perf, all_val_perf, all_test_perf = [None] * T, [None] * T, [None] * T

    time_lapsed = [None] * T
    D = random.randn(n_dims, n_dims)
    curr_task_range_tr = []  ##################
    # curr_task_range_tr = task_range_tr
    # for pure_task_idx, new_tr_task in enumerate([0.0]):
    for pure_task_idx, new_tr_task in enumerate(task_range_tr):  ##################
        t = time.time()
        W_pred = np.zeros((n_dims, n_tasks))
        curr_task_range_tr.append(new_tr_task)  ##################

        #####################################################
        # OPTIMISATION
        X_train, Y_train = [None] * n_tasks, [None] * n_tasks
        n_points = [0] * n_tasks
        for _, task_idx in enumerate(curr_task_range_tr):
            X_train[task_idx] = data['X_train'][task_idx]
            Y_train[task_idx] = data['Y_train'][task_idx]
            n_points[task_idx] = len(Y_train[task_idx])

        D = solve_wrt_D(D, training_settings, data, X_train, Y_train, n_points, curr_task_range_tr, param1)

        # D_vec = np.reshape(D, [np.size(D)])
        # batch_objective_vec(D_vec, X_train, Y_train, n_points, curr_task_range_tr)
        # batch_grad_vec(D_vec, data, curr_task_range_tr)
        #
        # batch_obj_cg = lambda D_vec: batch_objective_vec(D_vec, X_train, Y_train, n_points, curr_task_range_tr)
        # batch_grad_cg = lambda D_vec: batch_grad_vec(D_vec, data, curr_task_range_tr)
        # D_vec = fmin_cg(batch_obj_cg, x0=D_vec, maxiter=10**10, fprime=batch_grad_cg, gtol=10**-12)
        # D = np.reshape(D_vec, [n_dims, n_dims])
        #
        # # projection on the 1/lambda trace norm ball
        # U, s, Vt = svd(D)  # eigen
        # # U, s = np.linalg.eig(D)
        # s = np.sign(s) * [max(np.abs(s[i]) - param1, 0) for i in range(len(s))]
        # D = U @ np.diag(s) @ Vt

        time_lapsed[pure_task_idx] = time.time() - t
        # Check performance on ALL training tasks for this D
        W_pred = solve_wrt_w(D, data['X_train'], data['Y_train'], n_tasks, data, W_pred, task_range_tr)
        train_perf = mean_squared_error(data['X_train'], data['Y_train'], W_pred, task_range_tr)
        all_train_perf[pure_task_idx] = train_perf

        #####################################################
        # VALIDATION
        W_pred = solve_wrt_w(D, data['X_train'], data['Y_train'], n_tasks, data, W_pred, task_range_val)

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

        print("Batch LTL | best validation stats:")
        print('T: %3d | lambda: %6e | val MSE: %7.5f | test MSE: %7.5f | norm D: %4.2f' %
              (new_tr_task, param1, all_val_perf[pure_task_idx], all_test_perf[pure_task_idx], norm(D)))
        print("")

    results = {}
    # results['D'] = D
    results['all_train_perf'] = all_train_perf
    results['all_val_perf'] = all_val_perf
    results['all_test_perf'] = all_test_perf
    results['time_lapsed'] = time_lapsed

    save_results(results, data_settings, training_settings, filename, foldername)

    print("best validation stats:")
    print('Batch LTL: param1: %8e | val MSE: %7.5f | test MSE: %7.5f | norm D: %4.2f' %
          (param1, all_val_perf[pure_task_idx], all_test_perf[pure_task_idx], norm(D)))


def online_LTL(data, data_settings, training_settings):
    foldername = training_settings['foldername']
    filename = training_settings['filename']
    param1_range = training_settings['param1_range']
    n_tasks = data_settings['n_tasks']
    n_dims = data_settings['n_dims']
    task_range = data_settings['task_range']
    task_range_tr = data_settings['task_range_tr']
    task_range_val = data_settings['task_range_val']
    task_range_test = data_settings['task_range_test']

    T = len(task_range_tr)
    all_train_perf, all_val_perf, all_test_perf = [[] for i in range(T)], [[] for i in range(T)], [[] for i in range(T)]
    best_param, best_train_perf, best_val_perf, best_test_perf = \
        [None] * T, [None] * T, [None] * T, [None] * T
    all_individual_tr_perf = [[] for i in range(T)]
    time_lapsed = [None] * T

    c_iter = 0
    best_D = random.randn(n_dims, n_dims)
    for pure_task_idx, curr_task_range_tr in enumerate(task_range_tr):
        t = time.time()

        best_val_performance = -10 ** 8
        for param1_idx, param1 in enumerate(param1_range):
            W_pred = np.zeros((n_dims, n_tasks))

            D = best_D
            #####################################################
            # OPTIMISATION
            X_train, Y_train = [None] * n_tasks, [None] * n_tasks
            n_points = [0] * n_tasks
            for _, task_idx in enumerate([curr_task_range_tr]):
                X_train[task_idx] = data['X_train'][task_idx]
                Y_train[task_idx] = data['Y_train'][task_idx]
                n_points[task_idx] = len(Y_train[task_idx])

            D, c_iter = solve_wrt_D_stochastic(D, training_settings, data, X_train, Y_train, n_points,
                                               [curr_task_range_tr], param1, c_iter)

            time_lapsed[pure_task_idx] = time.time() - t

            # Check performance on ALL training tasks for this D
            W_pred = solve_wrt_w(D, data['X_train'], data['Y_train'], n_tasks, data, W_pred, task_range_tr)
            train_perf = mean_squared_error(data['X_train'], data['Y_train'], W_pred, task_range_tr)
            all_train_perf[pure_task_idx].append(train_perf)

            individual_tr_perf = [None] * len(task_range_tr)
            for idx, task_idx in enumerate(task_range_tr):
                individual_tr_perf[idx] = mean_squared_error(data['X_train'], data['Y_train'], W_pred, [task_idx])

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

            if val_perf > best_val_performance:
                best_val_performance = val_perf

                best_param[pure_task_idx] = param1
                best_D = D
                best_train_perf[pure_task_idx] = train_perf
                best_val_perf[pure_task_idx] = val_perf
                best_test_perf[pure_task_idx] = test_perf
                all_individual_tr_perf[pure_task_idx] = individual_tr_perf

        print("best validation stats:")
        print('T: %3d | lambda: %6e | val MSE: %7.5f | test MSE: %7.5f | norm D: %4.2f' %
              (pure_task_idx, best_param[pure_task_idx], best_val_perf[pure_task_idx], best_test_perf[pure_task_idx],
               norm(best_D)))
        print("")

    results = {}
    results['best_param'] = best_param
    # results['D'] = best_D
    results['best_train_perf'] = best_train_perf
    results['best_val_perf'] = best_val_perf
    results['best_test_perf'] = best_test_perf
    results['all_individual_tr_perf'] = all_individual_tr_perf
    results['time_lapsed'] = time_lapsed

    save_results(results, data_settings, training_settings, filename, foldername)