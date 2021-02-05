from src.post.read_data import read_data
from src.post.plots import plotError, plotGrid, plotImportance2D
import numpy as np

conditions = ['naive', 'itl', 'single_task', 'meta']
evaluations = ['MAE','MSE','MCA', 'CD']

skip = 1 # Plots look better if we skip the first one
data, weight, reg_par, pct_steps, nSeed, metaLen = read_data(evaluations, conditions,verbose=False)
plotError(data[:, :, :, :, skip:], pct_steps[skip:], conditions, evaluations, nSeed)
plotGrid(weight[:, :, :, :, skip:], pct_steps[skip:], metaLen)
wp = np.nanmean(weight, 1)
wp = np.nanmean(wp, 3)
wp = wp[:, :, [0, 129]]
plotImportance2D(wp, folder='../../analysis/', name='all_subj_Importance_2D.png')