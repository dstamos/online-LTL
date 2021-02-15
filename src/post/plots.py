import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import axes3d
import imageio
from glob import glob
from scipy import stats as sp
import numpy as np
import mne


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
    plt.plot(x, m, '.-', color=color, label=label)
    plt.fill_between(x, ll, hl, alpha=0.1, color=color)
    return


def plotError(data, settings, cond_to_plot, eval_to_plot, ratio, skip, title, folder='../../analysis/'):
    colors = ['tab:grey', 'tab:red', 'tab:green', 'tab:blue']
    f = plt.figure(figsize=figsize, dpi=dpi)
    c = 1
    for ev in eval_to_plot:
        plt.subplot(ratio[0], ratio[1], c)
        evi = settings['evaluations'].index(ev)
        for ind, nam in cond_to_plot.items():
            plotSE(data[evi, ind, :, skip:], settings['tr_pct'][skip:], colors[ind], nam, True, len(settings['seed_range']))
            plt.xlim([settings['tr_pct'][skip:][0], settings['tr_pct'][-1]])
        plt.ylabel(ev)
        c +=1
    plt.suptitle(title)
    plt.legend(frameon=False)
    plt.savefig(folder+title, pad_inches=0)
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

def plot_topos(w, folder='../../analysis/', name='all_subj_Scalp.png',supTitle=[]):
    vmin = np.min(w)
    vmax = np.max(w)
    f = plt.figure(figsize=figsize, dpi=dpi)
    channels = mne.channels.make_standard_montage('biosemi64').get_positions()['ch_pos']
    pos = np.array([channels[c][:2] for c in channels])
    titles1 = ['Not Merged', 'Merged']
    titles2 = ['Response Features', 'Stimulus Features']
    f, ax = plt.subplots(2, 2, figsize=figsize, dpi=dpi, sharex=True,sharey=True)
    for i in range(2):
        for j in range(2):
            mne.viz.plot_topomap(w[i, j*64:(j+1)*64], pos, vmin, vmax, 'coolwarm', axes=ax[i, j], show=False)
            if not supTitle:
                ax[i, j].set_title(titles1[i]+' - '+titles2[j])
    if supTitle:
        plt.suptitle(supTitle)
    plt.savefig(folder + name, pad_inches=0)
    plt.close(f)

def plot_topo_gif(w, folder='../../analysis/TopoGif/', name='all_subj_Scalp'):
    for i in range(w.shape[1]):
        ind = str(i).zfill(2)
        plot_topos(w[:, i, :], folder, name+ind+'.png','# Training sets: '+ind)
    files = sorted(glob(folder+'*.png'))
    images = []
    for f in files:
        images.append(imageio.imread(f))
    imageio.mimsave(folder+name+'.gif', images)
