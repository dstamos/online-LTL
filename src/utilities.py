from sklearn.metrics import mean_squared_error
import numpy as np
import pickle
import os


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
        performance = np.mean(performances)
        return performance
    else:
        all_performances = []
        for metamodel_idx in range(len(all_predictions)):
            metamodel_performances = []
            for task_idx in range(len(all_true_labels)):
                curr_perf = mae_clip(all_true_labels[task_idx], all_predictions[metamodel_idx][task_idx])
                metamodel_performances.append(curr_perf)
            curr_metamodel_performance = np.mean(metamodel_performances)
            all_performances.append(curr_metamodel_performance)
            # TODO Recover individual errors for each task as well. This way it can be investigate how the errors progress for each task
        return np.array(all_performances)


def multiple_tasks_evaluation(all_true_labels, all_predictions, all_correct, method):
    all_performances = []
    for metamodel_idx in range(len(all_predictions)):
        metamodel_performances = []
        for task_idx in range(len(all_true_labels)):
            curr_perf = evaluation_methods(all_true_labels[task_idx], all_predictions[metamodel_idx][task_idx], all_correct[task_idx], method)
            metamodel_performances.append(curr_perf)
        curr_metamodel_performance = np.mean(metamodel_performances, 0)
        all_performances.append(curr_metamodel_performance)
    return np.array(all_performances)


def mae_clip(labels, predictions):
    return np.median(np.abs(labels - np.clip(predictions, 0.1, 1)))


def mse_clip(labels, predictions):
    return np.mean((labels - np.clip(predictions, 0.1, 1)) ** 2)


def mca_clip(predictions, corr):
    return np.mean(1 - abs(np.clip(predictions, 0, 1) - corr))


def cd_clip(predictions, corr):
    pred = np.clip(predictions, 0, 1)
    if len(corr) == np.sum(corr):
        return -0.99
    return np.mean(pred[corr]) - np.mean(pred[~corr])


def evaluation_methods(labels, predictions, correct, method):
    res = []
    if 'MAE' in method:
        res.append(mae_clip(labels, predictions))
    if 'MSE' in method:
        res.append(mse_clip(labels, predictions))
    if 'MCA' in method:
        res.append(mca_clip(predictions, correct))
    if 'CD' in method:
        res.append(cd_clip(predictions, correct))
    return res


def save_results(results, foldername='results', filename='temp'):
    os.makedirs(foldername, exist_ok=True)
    filename = './' + foldername + '/' + filename + '.pckl'
    pickle.dump(results, open(filename, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
