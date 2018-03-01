import numpy as np

from utilities import synthetic_data_gen, mean_squared_error, batch_grad_func
from numpy.linalg import pinv
from numpy import identity as eye
import numpy.random as random
from numpy.linalg import norm
import matplotlib.pyplot as plt


np.random.seed(999)

# data generation
from fixed_run_settings import *
data = synthetic_data_gen(DATA_SETTINGS)

param1_range = ['param1_range']
n_tasks = DATA_SETTINGS['n_tasks']
n_dims = DATA_SETTINGS['n_dims']
Wtrue = data['W_true']
task_range = DATA_SETTINGS['task_range']
task_range_tr = DATA_SETTINGS['task_range_tr']
task_range_val = DATA_SETTINGS['task_range_val']
task_range_test = DATA_SETTINGS['task_range_test']

# optimisation

plt.figure()

param1 = 0.01

D = random.randn(n_dims, n_dims)

W_pred = np.zeros((n_dims, n_tasks))
Y_train_pred, Y_val_pred, Y_test_pred = [None] * n_tasks, [None] * n_tasks, [None] * n_tasks

X_train, Y_train = [None] * n_tasks, [None] * n_tasks
for _, task_idx in enumerate(task_range_tr):
    X_train[task_idx] = np.concatenate((data['X_val'][task_idx], data['X_train'][task_idx]))
    Y_train[task_idx] = np.concatenate((data['Y_val'][task_idx], data['Y_train'][task_idx]))

for _, task_idx in enumerate(task_range_tr):
    # fixing D and solving for w_t
    n_points = len(Y_train[task_idx])
    curr_w_pred = (
    D @ X_train[task_idx].T @ pinv(X_train[task_idx] @ D @ X_train[task_idx].T + n_points * eye(n_points)) @
    Y_train[task_idx]).ravel()
    W_pred[:, task_idx] = curr_w_pred

batch_objective = lambda D: sum([n_points * norm(pinv(X_train[i] @ D @ X_train[i].T + n_points * eye(n_points)) @ Y_train[i]) ** 2 for i in task_range_tr])
batch_grad = lambda D: batch_grad_func(D, task_range_tr, data)

Lipschitz = (6 / (np.sqrt(n_points) * n_points**2)) * max([norm(X_train[i], ord=np.inf)**3 for i in task_range_tr])
# Lipschitz = (2 / n_points) * max([norm(X_train[i], ord=np.inf) ** 2 for i in task_range_tr])
step_size = 1 / Lipschitz
# step_size = 55

curr_obj = batch_objective(D)

objectives = []
n_iter = 10 ** 10
curr_tol = 10 ** 10
conv_tol = 10 ** -6
c_iter = 0

while (c_iter < n_iter) and (curr_tol > conv_tol):
    prev_D = D
    prev_obj = curr_obj

    D = prev_D - step_size * batch_grad(prev_D)

    curr_obj = batch_objective(D)
    objectives.append(curr_obj)

    curr_tol = abs(curr_obj - prev_obj) / prev_obj

    if (c_iter % 2000 == 0):
        plt.plot(objectives, "b")
        plt.pause(0.0001)
        print("iter: %5d | obj: %20.18f | tol: %20.18f" % (c_iter, curr_obj, curr_tol))
    c_iter = c_iter + 1
a=1

