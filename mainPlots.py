from src.read_data import read_data
from src.post.plots import plotError, plotGrid, plotImportance2D, plot_change_distribution, plotWeight
from src.data_management_essex import load_data_essex_two
from src.utilities import fisher_clip
import numpy as np

settings = {'foldername': 'results',
            'seed_range': range(30),
            'tr_pct': np.arange(0.0, 0.825, 0.025),
            'prec': '-tr_pct_{:0.4f}',
            'evaluations': ['NMSE', 'FI', 'COR'],
            'conditions': ['naive', 'itl', 'single_task', 'meta'],
            'merged': [False],
            'nFeatures': 121,
            'dataset': 2,
            'useRT': False}
skip = 1
extra_eval = []
for fitness in ['NMSE']:
    settings['fitness'] = [fitness]
    data, ltl_w, weight, importance = read_data(settings, extra_eval, True)
    d1, d2, d3, d4, d5 = weight.shape
    areas = np.zeros((d1, d2, d3, 7, d5))
    areas[:, :, :, 0, :] = weight[:, :, :, 0, :]
    for i in range(6):
        areas[:, :, :, i + 1, :] = np.nansum(weight[:, :, :, i * 20 + 1:(i + 1) * 20 + 1, :], 3)

    cond_to_plot = {0: 'Independent',
                    1: 'Single',
                    2: 'Meta'}
    ratio = [2, 4]
    regions = ['Bias', 'Left', 'Right', 'Central', 'Occipital', 'Vertical', 'Horizontal']
    #title = 'Not Merged Weight Regions ' + fitness
    #plotWeight(areas[1], settings, cond_to_plot, regions, ratio, skip, title)
    title = 'Weight Regions ' + fitness
    plotWeight(areas[0], settings, cond_to_plot, regions, ratio, skip, title)
    title = 'Weight Change' + fitness
    plot_change_distribution(weight[0, :, 1:, :, skip:], settings, title=title, clip=[0.1, 1])
    #title = 'Not Merged Weight Change' + fitness
    #plot_change_distribution(weight[1, :, 1:, :, skip:], settings, title=title, clip=0.1)
    ratio = [1, 3]
    title = 'NMSE'
    cond_to_plot = {0: 'Naive',
                    1: 'Independent',
                    2: 'Single',
                    3: 'Meta'}
    eval_to_plot = ['NMSE', 'FI', 'COR']
    plotError(data[0], settings, cond_to_plot, eval_to_plot, ratio, skip, title)
    #title = 'Not Merged Fitness ' + fitness
    #plotError(data[1], settings, cond_to_plot, eval_to_plot, ratio, skip, title)
if False:
    settings['evaluations'] += extra_eval


    skip = 1


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
        title = 'Merged ' + fitness+' Subj' + str(i)
        plotError(data[0, :, :, i*30:(i+1)*30, :], settings, cond_to_plot, eval_to_plot, ratio, skip, title,xline=[[], [fcd[i]], []])
        title = 'Not Merged ' + fitness+' Subj' + str(i)
        plotError(data[1, :, :, i * 30:(i + 1) * 30, :], settings, cond_to_plot, eval_to_plot, ratio, skip, title,
                  xline=[[], [fcd[i]], []])

    temp = np.zeros((2, 3, 4, 10, 33))
    for i in range(10):
        temp[:, :, :, i, :] = np.nanmean(data[:, :, :, i * 30:(i + 1) * 30, :], 3)
    title = 'Merged ' + fitness+' average'
    plotError(temp[0, :, :, :, :], settings, cond_to_plot, eval_to_plot, ratio, skip, title)
    title = 'Not Merged ' + fitness+' average'
    plotError(temp[1, :, :, :, :], settings, cond_to_plot, eval_to_plot, ratio, skip, title)

    del data, temp

    plot_change_distribution(weight[0, :, 1:, :, skip:])


    title = 'Bias Weight - Grid'
    wp = weight[0, :, :, 0, skip:]
    w = np.nanmean(wp, 0)
    plotGrid(w, settings['tr_pct'][skip:], title)
    w2D = wp[:, -1, :]
    title = 'Bias Weight'
    plotImportance2D(w2D, settings, skip, title)
    title = 'Regularization Parameter'
    plotImportance2D(reg_par[0, :, skip:], settings, skip, title, ylabel='Regularization')

