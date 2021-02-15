import os
import pickle
import numpy as np
from src.preprocessing import PreProcess
from src.utilities import evaluation_methods
from src.independent_learning import train_test_itl
from src.data_management_essex import load_data_essex_one, load_data_essex_two, split_data_essex

def naive(data):
    res = []
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
        res.append(test_predictions)
    return res

def meta(data, best_model_ltl, w):
    x_test_tasks = data['test_tasks_tr_features']
    y_test_tasks = data['test_tasks_tr_labels']
    corr_test_tasks = data['test_tasks_tr_corr']

    fine_tuning_test = np.all([len(y_test_tasks[task_idx]) > 5 for task_idx in range(len(y_test_tasks))])
    if fine_tuning_test:
        preprocessing = PreProcess(threshold_scaling=True, standard_scaling=True, inside_ball_scaling=True, add_bias=True)
    else:
        # In the case we don't have training data on the test tasks or we just don't want to fine-tune.
        preprocessing = PreProcess(threshold_scaling=False, standard_scaling=False, inside_ball_scaling=True, add_bias=True)
    tr_tasks_tr_features, tr_tasks_tr_labels, _ = preprocessing.transform(data['tr_tasks_tr_features'], data['tr_tasks_tr_labels'], data['tr_tasks_tr_corr'], fit=True, multiple_tasks=True)
    # Test
    if fine_tuning_test:
        _, _, _ = preprocessing.transform(x_test_tasks, y_test_tasks, corr_test_tasks, fit=True, multiple_tasks=True)

    test_tasks_test_features, test_tasks_test_labels, test_tasks_test_corr = preprocessing.transform(
        data['test_tasks_test_features'], data['test_tasks_test_labels'], data['test_tasks_test_corr'], fit=False,
        multiple_tasks=True)

    if fine_tuning_test:
        return best_model_ltl.predict(test_tasks_test_features, w)[0]
    else:
        return best_model_ltl.predict(test_tasks_test_features)[0]


def reconstruct(models, seed, dataset, useRT, settings, best_moldel_ltl=[], w=[]):
    np.random.seed(seed)
    if dataset == 1:
        all_features, all_labels, all_experiment_names, all_correct = load_data_essex_one(useRT=useRT)
    else:
        all_features, all_labels, all_experiment_names, all_correct = load_data_essex_two(useRT=useRT)
    data = split_data_essex(all_features, all_labels, all_experiment_names, settings, verbose=False, all_corr=all_correct)
    pred = []
    for m in models:
        if m == 'naive':
            pred.append(naive(data))
        elif m == 'itl':
            pred.append(train_test_itl(data, settings, True))
        elif m == 'meta':
            if w:
                pred.append(meta(data, best_moldel_ltl, [w[-1]]))
            else:
                pred.append(meta(data, best_moldel_ltl, None))
    return data['test_tasks_test_labels'], pred, data['test_tasks_test_corr']

def calculate_evaluation(evals, models, seed, dataset, useRT, settings, best_moldel_ltl=[], w=[]):
    label, pred, corr = reconstruct(models, seed, dataset, useRT, settings, best_moldel_ltl, w)
    val = np.empty((len(pred), len(pred[0]), len(evals)))
    for i in range(len(pred)):
        for j in range(len(pred[0])):
            val[i, j ,:] = np.array(evaluation_methods(label[j], pred[i][j], corr[j], evals))
    return np.mean(val, 1)

# Settings a dictionary that must contain:
# foldername: str
# seed_range: str
# tr_pct: np.array
# prec: str
# fitness: list of str
# evaluations: list of str
# conditions: list of str
# merged: list of Bool
# nFetatures: int
def read_data(settings, add_extra=[], verbose=True):
    foldername = settings['foldername']
    seed_range = settings['seed_range']
    tr_pct = settings['tr_pct']
    prec = settings['prec']
    fitness = settings['fitness']
    evaluations = settings['evaluations']
    conditions = settings['conditions']
    nFeatures = settings['nFeatures']
    merged = settings['merged']
    dataset = settings['dataset']
    useRT = settings['useRT']

    nPoints = len(tr_pct)
    subjFolder = os.listdir(foldername)
    nSubj = len(subjFolder)
    metaLen = (nSubj-2)*3

    beval = len(evaluations)
    bevalex = len(add_extra)

    data = np.empty((len(fitness), len(merged), beval+bevalex, len(conditions), nSubj, nPoints))
    weight = np.empty((len(fitness), len(merged), nSubj, metaLen, nFeatures, nPoints))
    reg_par = np.empty((len(fitness), len(merged), nSubj, nPoints))

    seeds = np.empty((len(seed_range), len(evaluations+add_extra), len(conditions), nPoints))
    seeds_w = np.empty((len(seed_range), metaLen, nFeatures, nPoints))
    seeds_r = np.empty((len(seed_range), nPoints))
    seeds[:] = np.nan

    for s in range(nSubj):
        for fc, fit in enumerate(fitness):
            for mc, m in enumerate(merged):
                for pc, p in enumerate(tr_pct):
                    for seed in seed_range:
                        try:
                            filename = './' + foldername + '/' + subjFolder[s] + '/seed_' + str(seed) + prec.format(p)\
                                       + '-merge_test_' + str(m) + '-fitness_' + fit + '.pckl'
                            f = open(filename, "rb")
                            aux = pickle.load(f)
                            f.close()
                            for i, cond in enumerate(conditions):
                                if cond == 'meta':
                                    seeds[seed, :beval, i, pc] = aux['test_performance_meta'][-1, :]
                                else:
                                    seeds[seed, :beval, i, pc] = aux['test_performance_' + cond]
                            mdl = aux['best_model_meta']
                            w = aux['all_weight_vectors_meta']
                            if add_extra:
                                temp = calculate_evaluation(add_extra, conditions, seed, dataset, useRT, aux['settings'], mdl, w)
                                seeds[seed, beval:, :, pc] = temp
                            if w:
                                w = np.mean(np.abs(np.array(w)), 1)
                                seeds_w[seed, :, :, pc] = w / np.sum(w, 1, keepdims=True)
                            seeds_r[seed, pc] = mdl.regularization_parameter
                        except Exception as e:
                            if verbose:
                                print('broken', s, m, p, seed)
                            continue
                seeds[seeds == -.99] = np.nan
                data[fc, mc, :, :, s, :] = np.nanmean(seeds, 0)
                weight[fc, mc, s, :, :, :] = np.nanmean(seeds_w, 0)
                reg_par[fc, mc, s, :] = np.nanmean(seeds_r, 0)
                seeds[:] = np.nan
                seeds_w[:] = np.nan

    return data, weight, reg_par