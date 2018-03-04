import numpy as np

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

DATA_SETTINGS = {}
DATA_SETTINGS['seed'] = 999
DATA_SETTINGS['n_points'] = 1050
DATA_SETTINGS['n_dims'] = 20

n_tasks = 30
DATA_SETTINGS['task_range'] = list(np.arange(0, n_tasks))
DATA_SETTINGS['n_tasks'] = len(DATA_SETTINGS['task_range'])
DATA_SETTINGS['task_range_tr'] = list(split(range(n_tasks), 3))[0]
DATA_SETTINGS['task_range_val'] = list(split(range(n_tasks), 3))[1]
DATA_SETTINGS['task_range_test'] = list(split(range(n_tasks), 3))[2]


DATA_SETTINGS['train_perc'] = 0.75
DATA_SETTINGS['val_perc'] = 0.25
# DATA_SETTINGS['noise'] = 0.2
DATA_SETTINGS['noise'] = 0.25


TRAINING_SETTINGS = {}
# TRAINING_SETTINGS['param1_range'] = [10**float(i) for i in np.arange(-6, 5, 0.5)]
TRAINING_SETTINGS['n_iter'] = 10 ** 8
TRAINING_SETTINGS['conv_tol'] = 10 ** -10

