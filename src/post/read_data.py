import numpy as np
import os
import pickle

def read_data(evaluations, conditions, source = 'validation',verbose=True):
    dataset = 1
    if source == 'Dimitri':
        if dataset==1:
            foldername = '../../results/results-first_dataset_all_metrics'
        else:
            foldername = '../../results/results-second_dataset_all_metrics'
        seed_range = range(10)
        test_tasks_tr_points_pct = np.linspace(0.00, 0.8, 30)
        prec = '-tr_pct_{:0.4f}'
    elif source=='Jacobo':
        foldername = 'results-first_dataset'
        seed_range = [0]
        test_tasks_tr_points_pct = np.linspace(0.00, 0.8, 41)
        prec = '-tr_pct_{:0.2f}'
    elif source == 'validation':
        if dataset == 1:
            foldername = '../../results/results-first_dataset_validation_corrected'
        else:
            foldername = '../../results/results-second_dataset_validation_corrected'
        seed_range =range(6)
        prec = '-tr_pct_{:0.4f}'
        test_tasks_tr_points_pct = np.linspace(0.00, 0.8, 41)


    nPoints = len(test_tasks_tr_points_pct)
    subjFolder = os.listdir(foldername)
    nSubj = len(subjFolder)
    nFeatures = 130
    metaLen = (nSubj-2)*3


    # merged:True/False, evaluations, conditions, subjects, points
    data = np.empty((2, len(evaluations), len(conditions), nSubj, nPoints))
    weight = np.empty((2, nSubj, metaLen, nFeatures, nPoints))
    reg_par = np.empty((2, nSubj, nPoints))

    seeds = np.empty((len(seed_range), len(evaluations), len(conditions), nPoints))
    seeds_w = np.empty((len(seed_range), metaLen, nFeatures, nPoints))
    seeds_r = np.empty((len(seed_range), nPoints))
    seeds[:] = np.nan

    fitness = 'MSE'

    for s in range(nSubj):
        for mc, m in enumerate([False, True]):
            for pc, p in enumerate(test_tasks_tr_points_pct):
                for seed in seed_range:
                    try:
                        filename = './' + foldername + '/' + subjFolder[s] + '/seed_' + str(seed) + prec.format(p)\
                                   + '-merge_test_' + str(m) + '-fitness_' + fitness + '.pckl'
                        f = open(filename, "rb")
                        aux = pickle.load(f)
                        f.close()
                        for i in range(len(conditions) - 1):
                            seeds[seed, :, i, pc] = aux['test_performance_' + conditions[i]]
                        seeds[seed, :, 3, pc] = aux['test_performance_meta'][-1, :]
                        mdl = aux['best_model_meta']
                        w = np.abs(mdl.all_metaparameters_)
                        seeds_w[seed, :, :, pc] = w / np.sum(w, 1, keepdims=True)
                        seeds_r[seed, pc] = mdl.regularization_parameter
                    except Exception as e:
                        if verbose:
                            print('broken', s, m, p, seed)
                        continue
            seeds[seeds == -.99] = np.nan
            data[mc, :, :, s, :] = np.nanmean(seeds, 0)
            weight[mc, s, :, :, :] = np.nanmean(seeds_w, 0)
            reg_par[mc, s, :] = np.nanmean(seeds_r, 0)
            seeds[:] = np.nan
            seeds_w[:] = np.nan

    return data, weight, reg_par, test_tasks_tr_points_pct, len(seed_range), metaLen

