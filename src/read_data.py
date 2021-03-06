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
        npred.append(np.mean(data['test_tasks_tr_labels']) + np.zeros(len(data['test_tasks_test_labels'][i])))
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

    beval = len(evaluations)
    bevalex = len(add_extra)

    data = np.zeros((len(merged), beval+bevalex, len(conditions), nSubj*nSeed, nPoints))
    weight = np.zeros((len(merged), nSubj*nSeed, metaLen, nFeatures, nPoints))
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
                        for i, cond in enumerate(conditions):
                            if cond == 'meta':
                                data[mc, :beval, i, s*nSeed+seed, pc] = aux['test_performance_meta'][-1, :]
                                if aux['all_weight_vectors_meta'] is not None:
                                    pred.append(aux['all_weight_vectors_meta'][-1])
                                else:
                                    if m:
                                        pred.append([])
                                    else:
                                        pred.append([[], [], []])
                            elif cond == 'naive':
                                data[mc, :beval, i, s * nSeed + seed, pc] = aux['test_performance_naive']
                                if m:
                                    pred.append([np.nan])
                                else:
                                    pred.append(([[np.nan], [np.nan], [np.nan]]))
                            else:
                                data[mc, :beval, i, s * nSeed + seed, pc] =aux['test_performance_' + cond]
                                pred.append(aux['all_weights_' + cond])
                        mdl = aux['best_model_meta']
                        w = aux['all_weight_vectors_meta']
                        if add_extra:
                            temp = calculate_evaluation(add_extra, seed, dataset, useRT, aux['settings'], pred)
                            data[mc, beval:, :, s * nSeed + seed, pc] = temp
                        if not np.any(w==None):
                            w = np.mean(np.abs(np.array(w)), 1)
                            weight[mc, s * nSeed + seed, :, :, pc] = w / np.nansum(w, 1, keepdims=True)
                        reg_par[mc, s * nSeed + seed, pc] = mdl.regularization_parameter
                    except Exception as e:
                        if verbose:
                            print(e, s, m, p, seed)
    return data, weight, reg_par