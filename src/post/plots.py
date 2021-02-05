import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import axes3d
from scipy import stats as sp
import numpy as np
font = {'size': 20}
matplotlib.rc('font', **font)
dpi = 100
figsize =(1920 / dpi, 1080 / dpi)

def confidence_interval(data, axis=0, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.nanmean(a, axis), sp.sem(a, axis)
    h = se * sp.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h

def standard_error(data, axis=0, correct=False, nSeed=1):
    m, se = np.nanmean(data, axis), sp.sem(data, axis)
    if correct:
        se = se*np.sqrt(len(m)) /np.sqrt(len(m)*nSeed)
    return m, m - se, m + se

def plotSE(data, x, color, label, correct=False, nSeed=1):
    m, ll, hl = standard_error(data, correct=correct, nSeed=1)
    plt.plot(x, m, color, label=label)
    plt.fill_between(x, ll, hl, alpha=0.1, color=color)
    return

def plotError(data, pct_steps, conditions, evaluations, nSeed, folder='../../analysis/', name='all_subj_SE.png'):
    colors = ['tab:grey', 'tab:red', 'tab:green', 'tab:blue']
    labels = ['Naive', 'ITL', 'Single', 'Meta']
    titles1 =['Non-merged', 'Merged']
    titles2 =['MSE', 'CD']
    f = plt.figure(figsize=figsize, dpi=dpi)
    for i in range(2):
        for evc, ev in enumerate([1, 3]):
            plt.subplot(2, 2, i*2+evc+1)
            for cond in range(len(conditions)):
                plotSE(data[i, ev, cond, :, :], pct_steps, colors[cond], labels[cond], True, nSeed)
                plt.xlim([0.02, 0.8])
            plt.title(titles1[i]+' - '+titles2[evc])
    plt.tight_layout()
    plt.legend(frameon=False)
    plt.savefig(folder+name, pad_inches=0)
    plt.close(f)

def plotGrid(weight, pct_steps, metaLen, folder='../../analysis/', name='all_subj_Importance_Grid.png'):
    X, Y = np.meshgrid(pct_steps, np.arange(metaLen)+1)
    wp = np.nanmean(weight, 1)
    f = plt.figure(figsize=figsize, dpi=dpi)
    titles1 = ['Not Merged', 'Merged']
    titles2 = ['Bias', 'RT']
    for i in range(2):
        for fc, feat in enumerate([0, -1]):
            ax = f.add_subplot(2, 2, i*2+fc+1, projection='3d')
            ax.plot_wireframe(X, Y, wp[i, :, feat, :])
            plt.title(titles1[i]+' - '+titles2[fc])
    plt.xlabel('Fine Tunning')
    plt.ylabel('#Train Tasks')
    plt.savefig(folder+name, pad_inches=0)
    plt.close(f)

def plotImportance2D(w, folder='../../analysis/', name='all_subj_Importance_2D.png'):
    f = plt.figure(figsize=figsize, dpi=dpi)
    titles1 = ['Not Merged', 'Merged']
    titles2 = ['Bias', 'RT']
    for i in range(2):
        for feat in range(2):
            ax = f.add_subplot(2, 2, i * 2 + feat + 1)
            ax.plot(w[i,:,feat])
            plt.title(titles1[i] + ' - ' + titles2[feat])
    plt.ylabel('Importance')
    plt.xlabel('# Training Tasks')
    plt.savefig(folder + name, pad_inches=0)
    plt.close(f)