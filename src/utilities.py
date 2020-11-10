from sklearn.metrics import mean_squared_error
import numpy as np


def multiple_tasks_mse(all_true_labels, all_predictions, error_progression=False):
    if error_progression is False:
        performances = []
        for idx in range(len(all_true_labels)):
            curr_perf = mean_squared_error(all_true_labels[idx], all_predictions[idx])
            performances.append(curr_perf)
        performance = np.mean(performances)
        return performance
    else:
        all_performances = []
        for metamodel_idx in range(len(all_predictions)):
            metamodel_performances = []
            for idx in range(len(all_true_labels)):
                curr_perf = mean_squared_error(all_true_labels[idx], all_predictions[metamodel_idx][idx])
                metamodel_performances.append(curr_perf)
            curr_metamodel_performance = np.mean(metamodel_performances)
            all_performances.append(curr_metamodel_performance)
        return all_performances
