import numpy as np
# import optimisation
import matplotlib.pylab as plt
import numpy.random as random
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from numpy.linalg import pinv
from numpy import identity as eye
from numpy.linalg import svd
from numpy.linalg import lstsq
from numpy.linalg import solve

import scipy as sp
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score


import os
import sys
import pickle
import time

def synthetic_data_gen(data_settings):
    n_tasks = data_settings['n_tasks']

    n_train_tasks = round(n_tasks * 0.75)
    n_val_tasks = n_tasks - n_train_tasks
    n_test_tasks = 100
    data_settings['task_range_tr'] = list(range(n_train_tasks))
    data_settings['task_range_val'] = list(range(n_train_tasks,n_tasks))
    data_settings['task_range_test'] = list(range(n_train_tasks+n_val_tasks, n_train_tasks+n_val_tasks+n_test_tasks))
    n_tasks = n_train_tasks+n_val_tasks+n_test_tasks
    data_settings['n_tasks'] = n_tasks

    data_settings['task_range'] = list(np.arange(0, n_tasks))


    data_settings['train_perc'] = 0.75
    data_settings['val_perc'] = 0.25
    data_settings['noise'] = 0.2

    n_dims = data_settings['n_dims']
    n_points = data_settings['n_points']
    train_perc= data_settings['train_perc']
    val_perc = data_settings['val_perc']
    noise = data_settings['noise']

    generation_mode = 'sparse_explicit'
    if generation_mode == 'sparse_explicit':
        sparsity = 25 # 8

        fixed_sparsity =  random.choice(np.arange(0, n_dims), sparsity, replace=False)
        # diagonal = np.zeros(n_dims)
        # diagonal[fixed_sparsity] = 1
        # D = np.zeros((n_dims, n_dims))
        # D[np.diag_indices_from(D)] = diagonal

        data = {}
        W_true = np.zeros((n_dims, n_tasks))
        X_train, Y_train = [None]*n_tasks, [None]*n_tasks
        X_val, Y_val = [None] * n_tasks, [None] * n_tasks
        X_test, Y_test = [None] * n_tasks, [None] * n_tasks
        for task_idx in range(0, n_tasks):
            # generating and normalizing the data
            features = random.randn(n_points, n_dims)
            features = features / norm(features, axis=1, keepdims=True)

            # generating and normalizing the weight vectors
            weight_vector = np.zeros((n_dims, 1))
            weight_vector[fixed_sparsity] = random.randn(sparsity, 1)
            weight_vector = (weight_vector / norm(weight_vector)).ravel()

            labels = features @ weight_vector + noise * random.randn(n_points)

            X_train_all, X_test[task_idx], Y_train_all, Y_test[task_idx] = train_test_split(features, labels, test_size=100)#, random_state = 42)
            X_train[task_idx], X_val[task_idx], Y_train[task_idx], Y_val[task_idx] = train_test_split(X_train_all, Y_train_all, test_size=val_perc)

            W_true[:, task_idx] =  weight_vector

    data['X_train'] = X_train
    data['Y_train'] = Y_train
    data['X_val'] = X_val
    data['Y_val'] = Y_val
    data['X_test'] = X_test
    data['Y_test'] = Y_test
    data['W_true'] = W_true
    return data


def schools_data_gen(data_settings):

    temp = sio.loadmat('schoolData.mat')

    all_data = temp['X'][0]
    all_labels = temp['Y'][0]

    n_tasks = len(all_data)
    task_range_tr = data_settings['task_range_tr']
    task_range_val = data_settings['task_range_val']
    task_range_test = data_settings['task_range_test']


    X_train, Y_train = [None] * n_tasks, [None] * n_tasks
    X_val, Y_val = [None] * n_tasks, [None] * n_tasks
    X_test, Y_test = [None] * n_tasks, [None] * n_tasks

    # training tasks:
    max_score = 0
    for task_idx in task_range_tr:
        X_train[task_idx] = all_data[task_idx].T
        Y_train[task_idx] = all_labels[task_idx]

        # X_train[task_idx] = X_train[task_idx] / norm(X_train[task_idx], axis=1, keepdims=True)

        Y_train[task_idx] = Y_train[task_idx].ravel()

        if max(Y_train[task_idx]) > max_score:
            max_score = max(Y_train[task_idx])

        #
        # miny = min(min of whatever)
        # maxy = max(max of whather)
        #
        # y in [ 0 1]
        # (y - miny) / (maxy - miny)


        X_val[task_idx] = []
        Y_val[task_idx] = []

        X_test[task_idx] = []
        Y_test[task_idx] = []
        # print('training tasks | n_points: %3d' % len(Y_train[task_idx]))

    for task_idx in task_range_tr:
        Y_train[task_idx] = Y_train[task_idx] / 100


    max_score = 0
    for task_idx in task_range_val:
        X_train[task_idx], X_val[task_idx], Y_train[task_idx], Y_val[task_idx] = train_test_split(all_data[task_idx].T,
                                                                                                  all_labels[task_idx],
                                                                                                  test_size=0.5)

        Y_train[task_idx] = Y_train[task_idx].ravel()
        Y_val[task_idx] = Y_val[task_idx].ravel()

        if max(Y_train[task_idx]) > max_score:
            max_score = max(Y_train[task_idx])

        # X_train[task_idx] = X_train[task_idx] / norm(X_train[task_idx], axis=1, keepdims=True)
        # X_val[task_idx] = X_val[task_idx] / norm(X_val[task_idx], axis=1, keepdims=True)

        X_test[task_idx] = []
        Y_test[task_idx] = []
        # print('validation tasks | n_points training: %3d | n_points validation: %3d' % (len(Y_train[task_idx]), len(Y_val[task_idx])))

    for task_idx in task_range_val:
        Y_train[task_idx] = Y_train[task_idx] / 100
        Y_val[task_idx] = Y_val[task_idx] / 100

    max_score = 0
    for task_idx in task_range_test:
        X_train_all, X_test[task_idx], Y_train_all, Y_test[task_idx] = train_test_split(all_data[task_idx].T,
                                                                                        all_labels[task_idx],
                                                                                        test_size=0.5)  # , random_state = 42)
        X_train[task_idx], X_val[task_idx], Y_train[task_idx], Y_val[task_idx] = train_test_split(X_train_all,
                                                                                                  Y_train_all,
                                                                                                  test_size=0.5)
        Y_train[task_idx] = Y_train[task_idx].ravel()
        Y_val[task_idx] = Y_val[task_idx].ravel()
        Y_test[task_idx] = Y_test[task_idx].ravel()

        if max(Y_train[task_idx]) > max_score:
            max_score = max(Y_train[task_idx])

        # X_train[task_idx] = X_train[task_idx] / norm(X_train[task_idx], axis=1, keepdims=True)
        # X_val[task_idx] = X_val[task_idx] / norm(X_val[task_idx], axis=1, keepdims=True)
        # X_test[task_idx] = X_test[task_idx] / norm(X_test[task_idx], axis=1, keepdims=True)

        # print('test tasks | n_points training: %3d | n_points validation: %3d | n_points test: %3d' % (
        # len(Y_train[task_idx]), len(Y_val[task_idx]), len(Y_test[task_idx])))

    for task_idx in task_range_test:
        Y_train[task_idx] = Y_train[task_idx] / 100
        Y_val[task_idx] = Y_val[task_idx] / 100
        Y_test[task_idx] = Y_test[task_idx] / 100

    data_settings['n_dims'] = X_train[0].shape[1]

    data = {}
    data['X_train'] = X_train
    data['Y_train'] = Y_train
    data['X_val'] = X_val
    data['Y_val'] = Y_val
    data['X_test'] = X_test
    data['Y_test'] = Y_test
    return data, data_settings


def mean_squared_error(X, true, W, task_indeces):
    n_tasks = len(task_indeces)
    explained_variance = 0
    mse = 0
    for _, task_idx in enumerate(task_indeces):
        n_points = len(true[task_idx])
        pred = X[task_idx] @ W[:, task_idx]
        mse = norm(true[task_idx].ravel() - pred)**2 / n_points

        # mse = mse + norm(true[task_idx].ravel() - pred) ** 2 / n_points
        # explained_variance = explained_variance +  100 * (1 - mse/np.var(true[task_idx]))
        explained_variance = explained_variance + explained_variance_score(true[task_idx].ravel(), pred)

    explained_variance = 100 * explained_variance / n_tasks
    mse = mse / n_tasks
    return explained_variance


def batch_grad_func(D, task_indeces, data, switch):
    X = [data['X_train'][i] for i in task_indeces]
    Y = [data['Y_train'][i] for i in task_indeces]
    n_dims = X[0].shape[1]

    M = lambda D, n, t: X[t] @ D @ X[t].T + n * eye(n)

    grad = np.zeros((n_dims, n_dims))
    for idx, _ in enumerate(task_indeces):
        n_points = len(Y[idx])

        # tt = time.time()
        # invM = pinv(M(D, n_points, idx))
        # invM = lstsq(X[idx] @ D @ X[idx].T + n_points * eye(n_points), eye(n_points), rcond=None)[0]

        invM = sp.linalg.inv(M(D, n_points, idx))

        # MD = M(D, n_points, idx)
        # print('potato computed: %6.3f' % (time.time() - tt))

        Y[idx] = np.reshape(Y[idx], [1, len(Y[idx])])
        curr_grad = X[idx].T @ invM @ ((Y[idx].T @ Y[idx]) @ invM + invM @ (Y[idx].T @ Y[idx])) @ invM @ X[idx]

        # YY = np.multiply.outer(Y[idx].ravel(), Y[idx].ravel())
        # curr_grad = X[idx].T @ invM @ (YY @ invM + invM @ YY) @ invM @ X[idx]

        curr_grad = -n_points * curr_grad

        # half_grad = X[idx].T @ np.linalg.solve(MD.T, np.linalg.solve(MD,YY).T ).T

        # curr_grad2 = -n_points  *(half_grad  + half_grad.T)

        if switch == 1:
            # Lipschitz = 6
            # Lipschitz = (6 / (np.sqrt(n_points) * n_points ** 2)) * norm(X[idx], ord=2) ** 3

            Lipschitz = (6 / n_points) * norm(X[idx], ord=2) ** 3
            step_size  = 1 / Lipschitz
            grad = grad + step_size * curr_grad
        else:
            grad = grad + curr_grad
    return grad


def solve_wrt_w(D, X, Y, n_tasks, data, W_pred, task_range):
    for _, task_idx in enumerate(task_range):
        n_points = len(Y[task_idx])
        # replace pinv with np.linalg.solve or wahtever
        curr_w_pred = (D @ X[task_idx].T @ pinv(X[task_idx] @ D @ X[task_idx].T + n_points * eye(n_points)) @ Y[task_idx]).ravel()
        W_pred[:, task_idx] = curr_w_pred
    return W_pred


def solve_wrt_D(D, training_settings, data, X_train, Y_train, n_points, task_range, param1):
    batch_objective = lambda D: sum([n_points[i] * norm(sp.linalg.inv(X_train[i] @ D @ X_train[i].T +
                                                             n_points[i] * eye(n_points[i])) @ Y_train[i]) ** 2 for i in task_range])
    # batch_objective = lambda D: sum([n_points[i] * norm(lstsq(X_train[i] @ D @ X_train[i].T + n_points[i] * eye(n_points[i]), Y_train[i], rcond=None)[0]) ** 2 for i in task_range])

    batch_grad = lambda D: batch_grad_func(D, task_range, data, 1)

    # n_points_lip = max(n_points)
    # Lipschitz = (6 / (np.sqrt(n_points_lip) * n_points_lip ** 2)) * max([norm(X_train[i], ord=np.inf) ** 3 for i in task_range])
    # Lipschitz = 6 / n_points_lip
    # Lipschitz = //
    # step_size = 1 / Lipschitz

    # input = {}
    # input['D'] = D
    # input['X_train'] = [X_train[i] for i in task_range]
    # input['Y_train'] = [Y_train[i] for i in task_range]
    # input['n_points'] = [n_points[i] for i in task_range]
    # input['task_range'] = task_range
    # input['param1'] = param1
    # sp.io.savemat('cvx_business.mat', input)



    curr_obj = batch_objective(D)

    objectives = []
    n_iter = 50000
    curr_tol = 10 ** 10
    conv_tol = training_settings['conv_tol']
    c_iter = 0

    t = time.time()
    while (c_iter < n_iter) and (curr_tol > conv_tol):
        prev_D = D
        prev_obj = curr_obj

        # tt = time.time()
        D = prev_D - batch_grad(prev_D)
        # print('grad computed %6.3f' % (time.time() - tt))

        # projection on the 1/lambda trace norm ball
        # U, s, Vt = svd(D)  # eigen
        # s = [max(si,0),1/param1) for si in s]
        # D = U @ np.diag(s) @ Vt

        s, U = np.linalg.eigh(D)
        # s = [min(max(si,0),1/param1) for si in s]
        s = [max(si, 0) for si in s]
        s = s/sum(s)*1/param1
        D = U @ np.diag(s) @ U.T


        curr_obj = batch_objective(D)
        objectives.append(curr_obj)

        curr_tol = abs(curr_obj - prev_obj) / prev_obj
        c_iter = c_iter + 1

        if (time.time() - t > 10):
            t = time.time()
            # plt.plot(objectives, "b")
            # plt.pause(0.0001)
            print("iter: %5d | obj: %20.18f | tol: %20.18f" % (c_iter, curr_obj, curr_tol))

    # plt.plot(objectives)
    # plt.pause(0.01)
    print("iter: %5d | obj: %20.18f | tol: %20.18f" % (c_iter, curr_obj, curr_tol))

    return D


def solve_wrt_D_stochastic(D, training_settings, data, X_train, Y_train, n_points, task_range, param1, c_iter):
    batch_objective = lambda D: sum([n_points[i] * norm(pinv(X_train[i] @ D @ X_train[i].T + n_points[i] * eye(n_points[i])) @ Y_train[i]) ** 2 for i in task_range])
    batch_grad = lambda D: batch_grad_func(D, task_range, data, 0)

    c_value = training_settings['c_value']

    curr_obj = batch_objective(D)

    objectives = []

    n_points_for_step = np.array(n_points).astype('float')
    n_points_for_step[n_points_for_step == 0] = np.nan
    n_iter = 2
    # n_iter = np.ceil(1000 / np.sqrt(np.nanmean(n_points_for_step)))

    curr_tol = 10 ** 10
    conv_tol = 10 ** -5
    inner_iter = 0

    t = time.time()
    while (inner_iter < n_iter) and (curr_tol > conv_tol):
        inner_iter = inner_iter + 1
        prev_D = D
        prev_obj = curr_obj

        c_iter = c_iter + inner_iter
        step_size = c_value / np.sqrt(c_iter)
        D = prev_D - step_size * batch_grad(prev_D)

        curr_obj = batch_objective(D)
        objectives.append(curr_obj)

        curr_tol = abs(curr_obj - prev_obj) / prev_obj

        if (time.time() - t > 111):
            t = time.time()
            # plt.plot(objectives, "b")
            # plt.pause(0.0001)
            print("iter: %5d | obj: %20.18f | tol: %20.18f" % (c_iter, curr_obj, curr_tol))

        # projection on the 1/lambda trace norm ball
        # U, s, Vt = svd(D)  # eigen
        # s = [min(si,1/param1) for si in s]
        # D = U @ np.diag(s) @ Vt

        s, U = np.linalg.eigh(D)
        s = [max(si,0) for si in s]
        s = s/sum(s)*1/param1
        D = U @ np.diag(s) @ U.T


    return D, c_iter


def save_results(results, data_settings, training_settings, filename, foldername):
    param1_range = training_settings['param1_range']

    if not os.path.exists(foldername):
        os.makedirs(foldername)
    f = open(foldername + '/' + filename + ".pckl", 'wb')
    pickle.dump(results, f)
    pickle.dump(data_settings, f)
    pickle.dump(training_settings, f)
    f.close()
