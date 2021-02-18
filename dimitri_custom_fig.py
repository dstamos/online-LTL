import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from src.post.plots import standard_error

def read_data(verbose=True):
    foldername = ['results/results-second_dataset_nmse/',
                  'results/results-second_dataset_nmse_christoph_classic_range/',
                  'results/results-second_dataset_nmse_christoph_classic_range_0pct/',
                  'results/results-second_dataset_nmse_christoph_classic_range_0pct_norm/']
    seed_range = np.arange(20)
    evaluations = ['MAE', 'NMSE', 'MSE', 'MCA', 'CD']
    conditions = ['naive', 'itl', 'single_task', 'meta']

    subjFolder = os.listdir(foldername[0])
    nSubj = len(subjFolder)

    beval = len(evaluations)

    data = np.zeros((len(foldername), beval, len(conditions), nSubj))
    seeds = np.zeros((len(seed_range), len(evaluations), len(conditions)))
    seeds[:] = np.nan
    for fc, fol in enumerate(foldername):
        for s in range(nSubj):
            for seed in seed_range:
                try:
                    filename = fol + subjFolder[s] + '/seed_' + str(seed) \
                               + '-tr_pct_0.0000-merge_test_False-fitness_NMSE.pckl'
                    f = open(filename, "rb")
                    aux = pickle.load(f)
                    f.close()
                    for i, cond in enumerate(conditions):
                        if cond == 'meta':
                            seeds[seed, :, i] = aux['test_performance_meta'][-1, :]
                        else:
                            seeds[seed, :, i] = aux['test_performance_' + cond]
                    if np.sum(seeds[seed, :, :]) > 100:
                        print('a')
                except Exception as e:
                    if verbose:
                        print('broken', fol, s, seed)
                    continue
        seeds[seeds == -.99] = np.nan
        data[fc, :, :, s] = np.nanmean(seeds, 0)
        seeds[:] = np.nan

    return data

data = read_data()


full_m, full_ll, full_hl = standard_error(data[:, [1, 4], :, :], 3, True, 20)
wd = 0.2
pos = np.arange(4)
off = np.array([-.3, -.1, .1, .3])

labels = ['NMSE', 'Clasic Range', '0pct', 'norm']
titles = ['NMSE', 'CD']
appr = ['naive', 'itl', 'single_task', 'meta']
dpi = 100
figsize =(1920 / dpi, 1080 / dpi)
plt.figure(figsize=figsize, dpi=dpi)
for j in range(2):
    plt.subplot(1, 2, j+1)
    for i in range(4):
        plt.bar(pos+off[i], full_m[i, j, :], wd, label=labels[i])
        for k in range(4):
            plt.plot([pos[k]+off[i], pos[k]+off[i]], [full_ll[i,j,k], full_hl[i,j,k]], 'k')
    plt.title(titles[j])
    plt.xticks(np.arange(4), appr)
plt.legend()
plt.show()
