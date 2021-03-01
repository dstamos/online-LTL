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

def standard_error(data, axis=0):
    m, se = np.nanmean(data, axis), sp.sem(data, axis)
    return m, m - se, m + se

def plotSE(data, x, color, label):
    m, ll, hl = standard_error(data)
    plt.plot(x, m, '.-', color=color, label=label)
    plt.fill_between(x, ll, hl, alpha=0.1, color=color)
    return


def plotError(data, settings, cond_to_plot, eval_to_plot, ratio, skip, title, folder='analysis/', ylim=[], xline=[]):
    colors = ['tab:grey', 'tab:red', 'tab:green', 'tab:blue']
    f = plt.figure(figsize=figsize, dpi=dpi)
    for c, ev in enumerate(eval_to_plot):
        plt.subplot(ratio[0], ratio[1], c+1)
        evi = settings['evaluations'].index(ev)
        for ind, nam in cond_to_plot.items():
            plotSE(data[evi, ind, :, skip:], settings['tr_pct'][skip:], colors[ind], nam)
        xlim = [settings['tr_pct'][skip:][0], settings['tr_pct'][-1]]
        plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim[c])
        if xline:
            if xline[c]:
                plt.plot([xlim[0], xlim[1]], [xline[c], xline[c]], '--k', label='Ground Truth')
        plt.ylabel(ev)
    plt.suptitle(title)
    plt.legend(frameon=False)
    plt.savefig(folder+title, pad_inches=0)
    plt.close(f)

def plotGrid(w, pct_steps, title, ratio=[1, 1], folder='analysis/'):
    metalen = w.shape[-2]
    if len(w.shape) == 2:
        n_fig = 1
        w = np.expand_dims(w, 1)
    else:
        n_fig = w.shape[1]
    X, Y = np.meshgrid(pct_steps, np.arange(metalen)+1)
    f = plt.figure(figsize=figsize, dpi=dpi)
    for i in range(n_fig):
        ax = f.add_subplot(ratio[0], ratio[1], i+1, projection='3d')
        ax.plot_wireframe(X, Y, w[:, i, :])
    plt.xlabel('Fine Tunning')
    plt.ylabel('#Train Tasks')
    plt.suptitle(title)
    plt.savefig(folder+title, pad_inches=0)
    plt.close(f)

def plotImportance2D(w, settings, skip, title, ratio=[1, 1], ylabel='Importance', folder='analysis/'):
    if len(w.shape) == 2:
        n_fig = 1
        w = np.expand_dims(w, 1)
    else:
        n_fig = w.shape[1]
    f = plt.figure(figsize=figsize, dpi=dpi)
    for i in range(n_fig):
        plt.subplot(ratio[0], ratio[1], i + 1)
        plotSE(w[:, i, :], settings['tr_pct'][skip:], 'b', '')
    plt.ylabel(ylabel)
    plt.xlabel('% Fine tuning')
    plt.suptitle(title)
    plt.savefig(folder + title, pad_inches=0)
    plt.close(f)

def plot_topos(w, folder='analysis/', name='all_subj_Scalp.png',supTitle=[]):
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

def plot_topo_gif(w, folder='analysis/TopoGif/', name='all_subj_Scalp'):
    for i in range(w.shape[1]):
        ind = str(i).zfill(2)
        plot_topos(w[:, i, :], folder, name+ind+'.png','# Training sets: '+ind)
    files = sorted(glob(folder+'*.png'))
    images = []
    for f in files:
        images.append(imageio.imread(f))
    imageio.mimsave(folder+name+'.gif', images)

def plot_change_distribution(w, folder='analysis/', title = 'all_subj_change_distribution'):
    nseed = w.shape[0]
    nsteps = w.shape[-1]
    res = np.empty((nseed, nsteps))
    for i in range(nseed):
        for j in range(nsteps):
            res[i, j] = np.linalg.norm(w[i, 1, :, j] - w[i, -1, :, j])
    f = plt.figure(figsize=figsize, dpi=dpi)
    plt.matshow(res, 0)
    plt.xlabel('Target %')
    plt.ylabel('Seeds')
    plt.colorbar()
    plt.savefig(folder + title, pad_inches=0)
    plt.close(f)
