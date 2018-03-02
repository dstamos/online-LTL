import numpy as np
# import optimisation
import matplotlib.pylab as plt
import numpy.random as random
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from numpy.linalg import pinv
from numpy import identity as eye
from numpy.linalg import svd

import os
import sys
import pickle
import time

def synthetic_data_gen(data_settings):
    n_dims = data_settings['n_dims']
    n_tasks = data_settings['n_tasks']
    n_points = data_settings['n_points']
    train_perc= data_settings['train_perc']
    val_perc = data_settings['val_perc']
    noise = data_settings['noise']

    generation_mode = 'sparse_explicit'
    if generation_mode == 'sparse_explicit':
        sparsity = 3

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

            X_train_all, X_test[task_idx], Y_train_all, Y_test[task_idx] = train_test_split(features, labels, test_size=1000)#, random_state = 42)
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


def mean_squared_error(X, true, W, task_indeces):
    n_tasks = len(task_indeces)
    mse = 0
    for _, task_idx in enumerate(task_indeces):
        n_points = len(true[task_idx])
        pred = X[task_idx] @ W[:, task_idx]
        mse = mse + norm(true[task_idx] - pred)**2 / n_points
    mse = mse / n_tasks
    return mse


# def mean_squared_error(true, pred, task_indeces):
#     n_tasks = len(task_indeces)
#     mse = 0
#     for _, task_idx in enumerate(task_indeces):
#         n_points = len(true[task_idx])
#         mse = mse + norm(true[task_idx] - pred[task_idx])**2 / n_points
#     mse = mse / n_tasks
#     return mse


def weight_vector_perf(Wtrue, Wpred, task_indeces):
    n_tasks = len(task_indeces)
    err = 0
    for _, task_idx in enumerate(task_indeces):
        err = err + norm(Wtrue[:, task_idx] - Wpred[:, task_idx])**2
    err = err / n_tasks
    return err


def batch_grad_func(D, task_indeces, data):
    X = [np.concatenate((data['X_val'][i], data['X_train'][i])) for i in task_indeces]
    Y = [np.concatenate((data['Y_val'][i], data['Y_train'][i])) for i in task_indeces]
    n_dims = X[0].shape[1]

    M = lambda D, n, t: X[t] @ D @ X[t].T + n * eye(n)

    grad = np.zeros((n_dims, n_dims))
    for idx, task_idx in enumerate(task_indeces):
        n_points = len(Y[idx])

        invM = pinv(M(D, n_points, idx))

        curr_grad = -n_points * X[idx].T @ invM @ ((Y[idx] @ Y[idx].T) * invM + invM * (Y[idx] @ Y[idx].T)) @ invM @ X[idx]
        grad = grad + curr_grad
        return grad


def solve_wrt_w(D, X, Y, n_tasks, data, W_pred, task_range):
    for _, task_idx in enumerate(task_range):
        n_points = len(Y[task_idx])
        # replace pinv with np.linalg.solve or wahtever
        curr_w_pred = (D @ X[task_idx].T @ pinv(X[task_idx] @ D @ X[task_idx].T + n_points * eye(n_points)) @ Y[task_idx]).ravel()
        W_pred[:, task_idx] = curr_w_pred
    return W_pred


def solve_wrt_D(D, data, X_train, Y_train, n_points, task_range, param1):
    batch_objective = lambda D: sum([n_points * norm(pinv(X_train[i] @ D @ X_train[i].T + n_points * eye(n_points)) @ Y_train[i]) ** 2 for i in task_range])
    batch_grad = lambda D: batch_grad_func(D, task_range, data)

    Lipschitz = (6 / (np.sqrt(n_points) * n_points ** 2)) * max([norm(X_train[i], ord=np.inf) ** 3 for i in task_range])
    step_size = 1 / Lipschitz

    curr_obj = batch_objective(D)

    objectives = []
    n_iter = 10 ** 10
    curr_tol = 10 ** 10
    conv_tol = 10 ** -5
    c_iter = 0

    t = time.time()
    while (c_iter < n_iter) and (curr_tol > conv_tol):
        prev_D = D
        prev_obj = curr_obj

        D = prev_D - step_size * batch_grad(prev_D)

        curr_obj = batch_objective(D)
        objectives.append(curr_obj)

        curr_tol = abs(curr_obj - prev_obj) / prev_obj

        if (time.time() - t > 30):
            t = time.time()
            # plt.plot(objectives, "b")
            # plt.pause(0.0001)
            print("iter: %5d | obj: %20.18f | tol: %20.18f" % (c_iter, curr_obj, curr_tol))
        c_iter = c_iter + 1
    # projection on the 1/lambda trace norm ball
    U, s, Vt = svd(D)  # eigen
    # U, s = np.linalg.eig(D)
    s = np.sign(s) * [max(np.abs(s[i]) - param1, 0) for i in range(len(s))]
    D = U @ np.diag(s) @ Vt
    # D = U @ np.diag(s) @ U.T
    return D


def solve_wrt_D_stochastic(D, data, X_train, Y_train, n_points, task_range, param1, c_iter):
    batch_objective = lambda D: sum([n_points * norm(pinv(X_train[i] @ D @ X_train[i].T + n_points * eye(n_points)) @ Y_train[i]) ** 2 for i in task_range])
    batch_grad = lambda D: batch_grad_func(D, task_range, data)



    curr_obj = batch_objective(D)

    objectives = []
    n_iter = 10
    curr_tol = 10 ** 10
    conv_tol = 10 ** -5
    inner_iter = 0

    t = time.time()
    while (inner_iter < n_iter) and (curr_tol > conv_tol):
        inner_iter = inner_iter + 1
        prev_D = D
        prev_obj = curr_obj

        c_iter = c_iter + inner_iter
        step_size = 1 / np.sqrt(c_iter)
        D = prev_D - step_size * batch_grad(prev_D)

        curr_obj = batch_objective(D)
        objectives.append(curr_obj)

        curr_tol = abs(curr_obj - prev_obj) / prev_obj

        if (time.time() - t > 30):
            t = time.time()
            # plt.plot(objectives, "b")
            # plt.pause(0.0001)
            print("iter: %5d | obj: %20.18f | tol: %20.18f" % (c_iter, curr_obj, curr_tol))
    # projection on the 1/lambda trace norm ball
    U, s, Vt = svd(D)  # eigen
    # U, s = np.linalg.eig(D)
    s = np.sign(s) * [max(np.abs(s[i]) - param1, 0) for i in range(len(s))]
    D = U @ np.diag(s) @ Vt
    # D = U @ np.diag(s) @ U.T


    return D, c_iter


def save_results(results, data_settings, training_settings, dataset):
    seed = data_settings['seed']
    param1_range = training_settings['param1_range']

    if len(param1_range) == 1:
        save_filename = "seed_" + str(seed) + "-param1_" + str(param1_range[0])
        save_foldername = 'results/' + dataset
        if not os.path.exists(save_foldername):
            os.makedirs(save_foldername)
        f = open(save_foldername + '/' + save_filename + ".pckl", 'wb')
        pickle.dump(results, f)
        pickle.dump(data_settings, f)
        pickle.dump(training_settings, f)
        f.close()