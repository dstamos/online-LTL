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


def multiple_tasks_mae_clip(all_true_labels, all_predictions, error_progression=False):
    if error_progression is False:
        all_predictions = all_predictions[-1]

        performances = []
        for task_idx in range(len(all_true_labels)):
            curr_perf = mae_clip(all_true_labels[task_idx], all_predictions[task_idx])
            performances.append(curr_perf)
        performance = np.median(performances)
        return performance
    else:
        all_performances = []
        for metamodel_idx in range(len(all_predictions)):
            metamodel_performances = []
            for task_idx in range(len(all_true_labels)):
                curr_perf = mae_clip(all_true_labels[task_idx], all_predictions[metamodel_idx][task_idx])
                metamodel_performances.append(curr_perf)
            curr_metamodel_performance = np.median(metamodel_performances)
            all_performances.append(curr_metamodel_performance)
            # TODO Recover individual errors for each task as well. This way it can be investigate how the errors progress for each task
        return np.array(all_performances)


def mae_clip(labels, predictions):
    return np.median(np.clip(np.abs(labels - predictions), 0, 1))
