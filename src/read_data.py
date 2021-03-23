import os
import pickle
import numpy as np
from src.utilities import evaluation_methods
from src.data_management_essex import load_data_essex_one, load_data_essex_two, split_data_essex

def reconstruct(seed, dataset, useRT, settings):
    np.random.seed(seed)
    if dataset == 1:
        all_features, all_labels, all_experiment_names, all_correct = load_data_essex_one(useRT=useRT)
    else:
        all_features, all_labels, all_experiment_names, all_correct = load_data_essex_two(useRT=useRT)
    data = split_data_essex(all_features, all_labels, all_experiment_names, settings, verbose=False, all_corr=all_correct)
    npred = []
    for i in range(len(data['test_tasks_indexes'])):
        npred.append(np.mean(data['test_tasks_tr_labels'][i]) + np.zeros(len(data['test_tasks_test_labels'][i])))
    return data['test_tasks_test_labels'], data['test_tasks_test_corr'], data['test_tasks_test_features'], npred

def calculate_evaluation(evals, seed, dataset, useRT, settings, w):
    label, corr, feat, naive = reconstruct(seed, dataset, useRT, settings)
    val = np.zeros((len(evals), len(w), len(w[0])))
    try:
        for i in range(len(w)):
                for j in range(len(w[0])):
                    if np.sum(np.isnan(w[i][j])) > 0:
                        if np.isnan(naive[j][0]):
                            pred = np.random.rand(len(label[j]))
                        else:
                            pred = naive[j]
                    elif w[i][j]==[]:
                        pred = np.random.rand(len(label[j]))
                    else:
                        pred = np.matmul(feat[j], w[i][j][1:]) + w[i][j][0]
                    val[:, i, j] = np.array(evaluation_methods(label[j], pred, corr[j], evals))
    except Exception as e:
        pass
    return np.nanmean(val, 2)


def calculate_importance(seed, dataset, useRT, settings, w):
    _, _, feat, _ = reconstruct(seed, dataset, useRT, settings)
    importance = np.zeros((len(w), 7))
    try:
        for i in range(len(w)):
            temp_imp = np.zeros((len(w[0]), 7))
            for j in range(len(w[0])):
                temp_pred = np.zeros((len(feat[j]), 7))
                if w[i][j]==[]:
                    temp_pred[:] = 0
                else:
                    for k in range(6):
                        temp_pred[:, k] = np.matmul(feat[j][:, k*20:(k+1)*20], w[i][j][k*20+1:(k+1)*20+1])
                    temp_pred[:, 6] = w[i][j][0]
                    temp_pred = np.abs(temp_pred)
                    temp_pred = temp_pred / np.sum(temp_pred, 1, keepdims=True)
                temp_imp[j] = np.mean(temp_pred, 0)
            importance[i] = np.mean(temp_imp, 0)
    except Exception as e:
        pass
    return importance

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
    fitness = settings['fitness'][0]
    evaluations = settings['evaluations']
    conditions = settings['conditions']
    nFeatures = settings['nFeatures']
    merged = settings['merged']
    dataset = settings['dataset']
    useRT = settings['useRT']

    nPoints = len(tr_pct)
    subjFolder = os.listdir(foldername)
    nSubj = len(subjFolder)
    nSeed = len(seed_range)
    metaLen = (nSubj-2)*3
    notNaive = np.sum([i != 'naive' for i in conditions])

    beval = len(evaluations)
    bevalex = len(add_extra)


    data = np.zeros((len(merged), beval+bevalex, len(conditions), nSubj*nSeed, nPoints))
    ltl_w = np.zeros((len(merged), nSubj*nSeed, metaLen, nFeatures, nPoints))
    weight = np.zeros((len(merged), notNaive, nSubj*nSeed, nFeatures, nPoints))
    importance = np.zeros((len(merged), notNaive, nSubj*nSeed, 7, nPoints))
    reg_par = np.zeros((len(merged), nSubj*nSeed, nPoints))

    for s in range(nSubj):
        for mc, m in enumerate(merged):
            for pc, p in enumerate(tr_pct):
                for seed in seed_range:
                    try:
                        filename = './' + foldername + '/' + subjFolder[s] + '/seed_' + str(seed) + prec.format(p)\
                                   + '-merge_test_' + str(m) + '-fitness_' + fitness + '.pckl'
                        f = open(filename, "rb")
                        aux = pickle.load(f)
                        f.close()
                        pred = []
                        imp_pred = []
                        cc = 0
                        for i, cond in enumerate(conditions):
                            if cond == 'meta':
                                data[mc, :beval, i, s*nSeed+seed, pc] = aux['test_performance_meta'][-1, :]
                                if aux['all_weight_vectors_meta'] is not None:
                                    w = aux['all_weight_vectors_meta'][-1]
                                    pred.append(w)
                                    imp_pred.append(w)
                                    if len(w) == 1:
                                        w = np.abs(w[0])
                                    else:
                                        w = np.mean(np.abs(np.array(w)), 0)
                                    weight[mc, cc, s * nSeed + seed, :, pc] = w / np.nansum(w, 0, keepdims=True)
                                else:
                                    if m:
                                        pred.append([])
                                        imp_pred.append([])
                                    else:
                                        pred.append([[], [], []])
                                        imp_pred.append([[], [], []])
                                cc += 1
                            elif cond == 'naive':
                                data[mc, :beval, i, s * nSeed + seed, pc] = aux['test_performance_naive']
                                if m:
                                    pred.append([np.nan])
                                else:
                                    pred.append(([[np.nan], [np.nan], [np.nan]]))
                            else:
                                data[mc, :beval, i, s * nSeed + seed, pc] =aux['test_performance_' + cond]
                                w = aux['all_weights_' + cond]
                                pred.append(w)
                                imp_pred.append(w)
                                if len(w) == 1:
                                    w = np.abs(w[0])
                                else:
                                    w = np.mean(np.abs(np.array(w)), 0)
                                weight[mc, cc, s * nSeed + seed, :, pc] = w / np.nansum(w, 0, keepdims=True)
                                cc += 1
                        w = aux['all_weight_vectors_meta']
                        importance[mc, :, s * nSeed + seed, :, pc] = calculate_importance(seed, dataset, useRT, aux['settings'], imp_pred)
                        if add_extra:
                            temp = calculate_evaluation(add_extra, seed, dataset, useRT, aux['settings'], pred)
                            data[mc, beval:, :, s * nSeed + seed, pc] = temp
                        if not np.any(w==None):
                            if len(w[0]) == 1:
                                w = np.abs(np.array(w)).squeeze()
                            else:
                                w = np.mean(np.abs(np.array(w)), 1)
                            ltl_w[mc, s * nSeed + seed, :, :, pc] = w / np.nansum(w, 1, keepdims=True)
                    except Exception as e:
                        if verbose:
                            print(e, s, m, p, seed)
    return data, ltl_w, weight, importance
