from src.read_data import read_data
from src.post.plots import plotError, plotGrid, plotImportance2D
from src.data_management_essex import load_data_essex_two
from src.utilities import fisher_clip
import numpy as np
settings = {'foldername': 'merged_results',
            'seed_range': range(9),
            'tr_pct': [1, 2],
            'prec': '-tr_pct_{:0.4f}',
            'fitness': ['NMSE'],
            'evaluations': ['NMSE','COR', 'FI'],
            'conditions': ['naive', 'itl', 'single_task', 'meta'],
            'merged': [True],
            'nFeatures': 121,
            'dataset': 2,
            'useRT': False}

settings = {'foldername': 'results',
            'seed_range': range(30),
            'tr_pct': np.arange(0.0, 0.825, 0.025),
            'prec': '-tr_pct_{:0.4f}',
            'fitness': ['NMSE'],
            'evaluations': ['NMSE', 'FI', 'COR'],
            'conditions': ['naive', 'itl', 'single_task', 'meta'],
            'merged': [False],
            'nFeatures': 121,
            'dataset': 2,
            'useRT': False}

extra_eval = []
data, ltl_w, weight, importance = read_data(settings, extra_eval, True)
settings['evaluations'] += extra_eval
cond_to_plot = {0: 'Naive',
                1: 'Independent',
                2: 'Single',
                3: 'Meta'}
eval_to_plot = ['NMSE',  'FI', 'COR']
skip = 1
ratio = [1, 3]
np.zeros((len(merged), beval + bevalex, len(conditions), nSubj * nSeed, nPoints))
nMerge, nEval, nCond, nTrials, nPoints = data.shape
nSeed = len(settings['seed_range'])
grandAv = np.zeros((nMerge, nEval, nCond, nTrials//nSeed, nPoints))
for i in range(nTrials//nSeed):
    grandAv[:, :, :, i, :] = np.mean(data[:, :, :, i*nSeed:(i+1)*nSeed, :], 3)
title = 'SE Gran Average'
plotError(grandAv[0, :, :, :, :], settings, cond_to_plot, eval_to_plot, ratio, skip, title)

if False:
    all_features, all_labels, all_experiment_names, all_correct = load_data_essex_two(useRT=False)
    fcd = np.zeros(10)
    for i in range(10):
        aux = np.zeros(3)
        for j in range(3):
            corr = all_correct[i*3+j]
            conf = all_labels[i*3+j]
            aux[j] = fisher_clip(conf, corr)
        fcd[i] = np.mean(aux)

    for i in range(10):
        title = 'SE for different Subj ' + str(i)
        plotError(data[0, :, :, i*30:(i+1)*30, :], settings, cond_to_plot, eval_to_plot, ratio, skip, title,xline=[[], [fcd[i]]])

    plot_change_distribution(w[0])


    title = 'Bias Weight - Grid'
    wp = weight[0, :, :, 0, skip:]
    w = np.nanmean(wp, 0)
    plotGrid(w, settings['tr_pct'][skip:], title)
    w2D = wp[:, -1, :]
    title = 'Bias Weight'
    plotImportance2D(w2D, settings, skip, title)
    title = 'Regularization Parameter'
    plotImportance2D(reg_par[0, :, skip:], settings, skip, title, ylabel='Regularization')

