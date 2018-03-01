import numpy as np
# import optimisation
import matplotlib.pylab as plt
import numpy.random as random
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from numpy.linalg import pinv
from numpy import identity as eye


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


def mean_squared_error(true, pred, task_indeces):
    n_tasks = len(task_indeces)
    mse = 0
    for _, task_idx in enumerate(task_indeces):
        n_points = len(true[task_idx])
        mse = mse + norm(true[task_idx] - pred[task_idx])**2 / n_points
    mse = mse / n_tasks
    return mse


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

        curr_grad = -n_points * X[idx].T @ invM @ ((Y[idx] @ Y[idx].T) * invM + invM * (Y[idx] @ Y[idx].T)) @ invM @ X[
            idx]
        grad = grad + curr_grad
        return grad


def parameter_selection(data, data_settings, training_settings, task_idx):
    param1_range = training_settings['param1_range']
    task_range = data_settings['task_range']
    n_tasks = task_range[task_idx]
    n_dims = data_settings['n_dims']

    WpredVARbatch = np.zeros((n_dims, n_tasks))
    for cLambdaIDX in range(0, len(param1_range)):
        cLambda = param1_range[cLambdaIDX]

        # batch version of VAR regularization
        training_settings['WpredVARbatch'] = WpredVARbatch
        WpredVARbatch, objectives = optimisation.gradientDescent(data, data_settings, training_settings, cLambda, n_tasks)
    outputs = {}
    outputs['WpredVARbatch'] = WpredVARbatch
    outputs['objectives'] = objectives
    return outputs