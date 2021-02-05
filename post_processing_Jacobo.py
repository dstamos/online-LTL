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
        se = se*np.sqrt(len(m)) /np.sqrt(len(m)*nSeed)
    return m, m - se, m + se

def plotSE(data, x, color, label, correct=False):
    m, ll, hl = standard_error(data, correct=correct)
    plt.plot(x, m, color, label=label)
    plt.fill_between(x, ll, hl, alpha=0.1, color=color)
    return



font = {'size': 20}
matplotlib.rc('font', **font)

source = 'validation'
dataset = 2
if source == 'Dimitri':
    if dataset==1:
        foldername = 'results/results-first_dataset_all_metrics'
    else:
        foldername = 'results/results-second_dataset_all_metrics'
    seed_range = range(10)
    nSeed = len(seed_range)
    test_tasks_tr_points_pct = np.linspace(0.00, 0.8, 30)
    prec = '-tr_pct_{:0.4f}'
    correct = True
elif source=='Jacobo':
    foldername = 'results-first_dataset'
    seed_range = [0]
    test_tasks_tr_points_pct = np.linspace(0.00, 0.8, 41)
    prec = '-tr_pct_{:0.2f}'
    correct = False
elif source == 'validation':
    if dataset == 1:
        foldername = 'results/results-first_dataset_validation_corrected'
    else:
        foldername = 'results/results-second_dataset_validation_corrected'
    seed_range =range(6)
    nSeed = len(seed_range)
    prec = '-tr_pct_{:0.4f}'
    test_tasks_tr_points_pct = np.linspace(0.00, 0.5, 26)
    correct = True


nPoints = len(test_tasks_tr_points_pct)
subjFolder = os.listdir(foldername)
nSubj = len(subjFolder)
conditions = ['naive', 'itl', 'single_task', 'meta']
evaluations = ['MAE','MSE','MCA', 'CD']
# merged:True/False, evaluations, conditions, subjects, points
data = np.empty((2, len(evaluations), len(conditions), nSubj, nPoints))
seeds = np.empty((len(seed_range), len(evaluations), len(conditions), nPoints))
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
                except Exception as e:
                    print('broken', s, m, p, seed)
                    continue
        seeds[seeds == -.99] = np.nan
        data[mc, :, :, s, :] = np.nanmean(seeds, 0)
        seeds[:] = np.nan

colors = ['tab:grey', 'tab:red', 'tab:green', 'tab:blue']
labels = ['Naive', 'ITL', 'Single', 'Meta']
dpi = 100

f = plt.figure(figsize=(1920 / dpi, 1080 / dpi), dpi=dpi)
for i in range(2):
    for evc, ev in enumerate([1, 3]):
        plt.subplot(2, 2, i*2+evc+1)
        for cond in range(len(conditions)):
            plotSE(data[i, ev, cond, :, :], test_tasks_tr_points_pct, colors[cond], labels[cond], correct)
            plt.xlim([0.0, 0.5])
        plt.ylabel(evaluations[ev])
plt.tight_layout()
plt.legend(frameon=False)
plt.savefig('analysis/all_subj_SE_'+source+str(dataset)+ '_' + fitness + '.png', pad_inches=0)
plt.close(f)
