import numpy as np
from numpy.linalg import pinv
from numpy import identity as eye
from src.preprocessing import PreProcess
from time import time
from sklearn.metrics import mean_squared_error
from src.utilities import mae_clip


def train_test_naive_transfer(data, settings):
    # Merge training/validation tasks (including training points from test tasks, if they exist)
    # Find average label
    # Apply on test tasks

    tt = time()
    all_performances = []
    for task_idx in range(len(data['test_tasks_indexes'])):
        y_tr = data['test_tasks_tr_labels'][task_idx]
        y_test = data['test_tasks_test_labels'][task_idx]

        if len(y_tr) > 1:
            prediction_value = np.mean(y_tr)
        else:
            # In the case we have no data for training (cold start), just use random data.
            prediction_value = np.random.uniform(0.1, 1.0, 1)

        # Testing
        test_predictions = prediction_value * np.ones(len(y_test))
        all_performances.append(mae_clip(y_test, test_predictions))
    test_performance = np.mean(all_performances)
    print(f'{"Naive":12s} | test performance: {test_performance:12.5f} | {time() - tt:5.2f}sec')

    return test_performance
