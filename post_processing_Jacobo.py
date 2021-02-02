import numpy as np
import scipy as sp
from scipy import stats as sp
import matplotlib.pyplot as plt
import matplotlib
import pickle
import os

def confidence_interval(data, axis=0, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.nanmean(a, axis), sp.sem(a, axis)
    h = se * sp.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h

def standard_error(data, axis=0, correct=False):
    m, se = np.nanmean(data, axis), sp.sem(data, axis)
    if correct:
        se = se*np.sqrt(len(m)) /np.sqrt(len(m)*10)
    return m, m - se, m + se

def plotSE(data, x, color, label, correct=False):
    m, ll, hl = standard_error(data, correct=correct)
    plt.plot(x, m, color, label=label)
    plt.fill_between(x, ll, hl, alpha=0.1, color=color)
    return



font = {'size': 20}
matplotlib.rc('font', **font)

source = 'Dimitri'
dataset = 2
if source == 'Dimitri':
    if dataset==1:
        foldername = 'results/results-first_dataset_all_metrics'
    else:
        foldername = 'results/results-second_dataset_all_metrics'
    seed_range = range(10)
    test_tasks_tr_points_pct = np.linspace(0.00, 0.8, 30)
    prec = '-tr_pct_{:0.4f}'
    correct = True
else:
    foldername = 'results-first_dataset'
    seed_range = [0]
    test_tasks_tr_points_pct = np.linspace(0.00, 0.8, 41)
    prec = '-tr_pct_{:0.2f}'
    correct = False

nPoints = len(test_tasks_tr_points_pct)
subjFolder = os.listdir(foldername)
nSubj = len(subjFolder)
conditions = ['naive', 'itl', 'single_task', 'meta']
evaluations = ['MAE', 'MSE', 'MCA', 'CD']
# merged:True/False, evaluations, conditions, subjects, points
data = np.empty((2, len(evaluations), len(conditions), nSubj, nPoints))
seeds = np.empty((len(seed_range), len(evaluations), len(conditions), nPoints))
seeds[:] = np.nan
for s in range(nSubj):
    for mc, m in enumerate([True, False]):
        for pc, p in enumerate(test_tasks_tr_points_pct):
            for seed in seed_range:
                try:
                    filename = './' + foldername + '/' + subjFolder[s] + '/seed_' + str(seed) + prec.format(p)\
                               + '-merge_test_' + str(m) + '.pckl'
                    f = open(filename, "rb")
                    aux = pickle.load(f)
                    f.close()
                    for i in range(len(conditions) - 1):
                        seeds[seed, :, i, pc] = aux['test_performance_' + conditions[i]]
                    seeds[seed, :, 3, pc] = aux['test_performance_meta'][-1, :]
                except Exception as e:
                    print('broken', e, seed)
                    continue
        data[mc, :, :, s, :] = np.nanmean(seeds, 0)
        seeds[:] = np.nan

colors = ['tab:grey', 'tab:red', 'tab:green', 'tab:blue']
labels = ['Naive', 'ITL', 'Single', 'Meta']
dpi = 100

for i in range(2):
    f = plt.figure(figsize=(1920 / dpi, 1080 / dpi), dpi=dpi)
    for ev in range(len(evaluations)):
        plt.subplot(2, 2, ev+1)
        for cond in range(len(conditions)):
            plotSE(data[i, ev, cond, :, :], test_tasks_tr_points_pct, colors[cond], labels[cond], correct)
            plt.xlim([0.02, 0.8])
        plt.ylabel(evaluations[ev])
    plt.tight_layout()
    plt.legend(frameon=False)
    plt.savefig('analysis/all_subj_SE_'+source+str(dataset)+'_merged_' + str(bool(i)) + '.png', pad_inches=0)
    plt.close(f)
k = 1
