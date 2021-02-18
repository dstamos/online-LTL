from src.read_data import read_data
from src.post.plots import plotError, plotGrid, plotImportance2D
import numpy as np

settings = {'foldername': 'results/results-second_dataset_nmse_christoph_new_data',
            'seed_range': range(4),
            'tr_pct': np.arange(0.0, 0.625, 0.025),
            'prec': '-tr_pct_{:0.4f}',
            'fitness': ['NMSE'],
            'evaluations': ['MAE', 'NMSE', 'MSE', 'MCA', 'CD'],
            'conditions': ['naive', 'itl', 'single_task', 'meta'],
            'merged': [False],
            'nFeatures': 145,
            'dataset': 2,
            'useRT': False}

data, weight, reg_par = read_data(settings, verbose=False)
cond_to_plot = {0: 'Naive',
                1: 'Independent',
                2: 'Single',
                3: 'Meta'}
eval_to_plot = ['NMSE', 'CD']
skip = 0
ratio = [1, 2]
title = 'SE for different Methods'
plotError(data[0, :, :, :, :], settings, cond_to_plot, eval_to_plot, ratio, skip, title)
title = 'Bias Weight - Grid'
wp = weight[0, :, :, 0, skip:]
w = np.nanmean(wp, 0)
plotGrid(w, settings['tr_pct'][skip:], title)
w2D = wp[:, -1, :]
title = 'Bias Weight'
plotImportance2D(w2D, settings, skip, title)
title = 'Regularization Parameter'
plotImportance2D(reg_par[0, :, skip:], settings, skip, title, ylabel='Regularization')

