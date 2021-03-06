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


def nmse_clip(labels, predictions):
    predictions = np.clip(predictions, 0.0, 1)
    mse = mean_squared_error(labels, predictions)
    if len(np.unique(labels)) == 1:
        return mse
    nmse = mse / mean_squared_error(labels.ravel(), np.mean(labels.ravel()) * np.ones(len(labels)))
    return nmse


def mca_clip(predictions, corr):
    return np.mean(1 - abs(np.clip(predictions, 0, 1) - corr))


def cd_clip(predictions, corr):
    pred = np.clip(predictions, 0, 1)
    if len(corr) == np.sum(corr):
        return np.nan
    return np.mean(pred[corr]) - np.mean(pred[~corr])

def ncd_clip(predictions, corr):
    pred = np.clip(predictions, 0, 1)
    if len(corr) == np.sum(corr):
        return np.nan
    if len(np.unique(pred[corr])) == 1 or len(np.unique(pred[~corr])) == 1:
        return 0
    return np.mean(pred[corr])/np.std(pred[corr]) - np.mean(pred[~corr])/np.std(pred[~corr])


def correlation_clip(labels, predictions):
    clipcor = np.clip(predictions, 0.1, 1)
    if len(set(clipcor)) == 1:
        return 0
    corr = np.corrcoef(labels, clipcor)
    return corr[0, 1]

def fisher_clip(predictions, corr):
    pred = np.clip(predictions, 0, 1)
    if len(corr) == np.sum(corr):
        return np.nan
    if len(np.unique(pred[corr])) == 1 or len(np.unique(pred[~corr])) == 1:
        return 0
    mc = np.mean(pred[corr])
    mi = np.mean(pred[~corr])
    m = np.mean(pred)
    vc = np.var(pred[corr])
    vi = np.var(pred[~corr])
    nc = np.sum(corr)
    ni = np.sum(~corr)
    return (nc*(mc-m)**2 + ni*(mi-m)**2) / (nc*vc + ni*vi)

def fisher_clip_equal(predictions, corr):
    pred = np.clip(predictions, 0, 1)
    if len(corr) == np.sum(corr):
        return np.nan
    if len(np.unique(pred[corr])) == 1 or len(np.unique(pred[~corr])) == 1:
        return 0
    return (np.mean(pred[corr])**2 - np.mean(pred[~corr])**2) / (np.var(pred[corr]) + np.var(pred[~corr]))

def evaluation_methods(labels, predictions, correct, methods):
    res = []
    for method in methods:
        if 'MAE' == method:
            res.append(mae_clip(labels, predictions))
        elif 'NMSE' == method:
            res.append(nmse_clip(labels, predictions))
        elif 'MSE' == method:
            res.append(mse_clip(labels, predictions))
        elif 'MCA' == method:
            res.append(mca_clip(predictions, correct))
        elif 'CD' == method:
            res.append(cd_clip(predictions, correct))
        elif 'COR' == method:
            res.append(correlation_clip(labels, predictions))
        elif 'NCD' == method:
            res.append(ncd_clip(predictions, correct))
        elif 'FI' == method:
            res.append(fisher_clip(predictions, correct))
        elif 'FIE' == method:
            res.append(fisher_clip_equal(predictions, correct))
    return res


def save_results(results, foldername='results', filename='temp'):
    os.makedirs(foldername, exist_ok=True)
    filename = './' + foldername + '/' + filename + '.pckl'
    pickle.dump(results, open(filename, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
