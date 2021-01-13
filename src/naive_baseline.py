import numpy as np
from numpy.linalg import pinv
from numpy import identity as eye
from src.preprocessing import PreProcess
from time import time
from sklearn.metrics import mean_squared_error
from src.utilities import mae_clip


def train_test_naive(data, settings):
    # "Training"
    tt = time()
    all_performances = []
    for task_idx in range(len(data['test_tasks_indexes'])):
        # First merge, since there is no validation in this naive approach.
        y_tr = np.concatenate([data['test_tasks_tr_labels'][task_idx], data['test_tasks_val_labels'][task_idx]])
        y_test = data['test_tasks_val_labels'][task_idx]

        prediction_value = np.mean(y_tr)

        # Testing
        test_predictions = prediction_value * np.ones(len(y_test))
        all_performances.append(mae_clip(y_test, test_predictions))
    test_performance = np.median(all_performances)
<<<<<<< HEAD
    print(f'{"Naive":12s} | test performance: {test_performance:12.5f} | {time() - tt:5.2f}sec')
=======
    print(f'Naive | test performance: {test_performance:12.5f} | {time() - tt:5.2f}sec')
>>>>>>> bc1940b02fdeb45596d3a6f230cec854d8338542

    return test_performance
