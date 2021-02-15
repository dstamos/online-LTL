from src.read_data import read_data
from src.post.plots import plotError
import numpy as np

settings = {'foldername': '../../results/results-first_dataset_nmse_30_seeds',
            'seed_range': range(30),
            'tr_pct': np.arange(0.0, 0.625, 0.025),
            'prec': '-tr_pct_{:0.4f}',
            'fitness': ['NMSE'],
            'evaluations': ['MAE', 'NMSE', 'MSE', 'MCA', 'CD'],
            'conditions': ['naive', 'itl', 'meta'],
            'merged': [False],
            'nFeatures': 129,
            'dataset': 1,
            'useRT': False}

data, weight, reg_par = read_data(settings, verbose=False)
cond_to_plot = {0: 'Naive',
                1: 'Independent',
                3: 'Meta'}
eval_to_plot = ['NMSE', 'CD']
skip = 1
ratio = [1, 2]
title = 'SE for different Methods'
plotError(data[0, 0, :, :, :, :], settings, cond_to_plot, eval_to_plot, ratio, skip, title)
if False:
    plotGrid(weight[:, :, :, :, skip:], pct_steps[skip:], metaLen)
    wp = np.nanmean(weight, 1)[:, :, [0, 129], -1]
    plotImportance2D(wp)
    w = np.nanmean(weight, 1)[:, -1, 1:-1, -1]
    plot_topos(w)
