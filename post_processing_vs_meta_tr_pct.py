import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import os
import time
import matplotlib.ticker as mtick
from scipy.stats import sem

font = {'size': 36}
matplotlib.rc('font', **font)

seed_range = range(0, 30)
test_subject_range = range(10)
merge_test = False
validation_evaluation_idx = 0
plot_evaluation_idx = 0
tr_pct_range = np.arange(0.05, 1.05, 0.05)

evaluation_names = ['NMSE', 'FI', 'COR']
foldername = 'results-meta_training_investigation'

all_errors_itl = []
all_errors_naive = []

all_subjects_naive = np.full([len(test_subject_range), len(tr_pct_range)], np.nan)
all_subjects_single_task = np.full([len(test_subject_range), len(tr_pct_range)], np.nan)
all_subjects_meta = np.full([len(test_subject_range), len(tr_pct_range)], np.nan)
all_subjects_itl = np.full([len(test_subject_range), len(tr_pct_range)], np.nan)

for test_subject_idx, test_subject in enumerate(test_subject_range):
    all_seeds_naive = np.full([len(seed_range), len(tr_pct_range)], np.nan)
    all_seeds_single_task = np.full([len(seed_range), len(tr_pct_range)], np.nan)
    all_seeds_meta = np.full([len(seed_range), len(tr_pct_range)], np.nan)
    all_seeds_itl = np.full([len(seed_range), len(tr_pct_range)], np.nan)
    for seed_idx, seed in enumerate(seed_range):
        foldername_with_subfolder = foldername + '/test_subject_' + str(test_subject)

        for tr_pct_idx, tr_pct in enumerate(tr_pct_range):

            filename = './' + foldername_with_subfolder + '/' + 'seed_' + str(seed) + '-meta_tr_pct_{:0.4f}'.format(tr_pct)+'-merge_test_'+str(merge_test) + '-fitness_' + evaluation_names[validation_evaluation_idx] + '.pckl'

            try:
                results = pickle.load(open(filename, "rb"))
                test_performance_naive = results['test_performance_naive'][plot_evaluation_idx]
                test_performance_single_task = results['test_performance_single_task'][plot_evaluation_idx]
                test_performance_itl = results['test_performance_itl'][plot_evaluation_idx]
                test_performance_meta = results['test_performance_meta']
                all_weight_vectors_meta = results['all_weight_vectors_meta']
                best_model_meta = results['best_model_meta']
                settings = results['settings']

            except Exception as e:
                print(f'failed: subject: {test_subject:2d} | seed: {seed:2d} | pct: {tr_pct:0.4f}')
                continue

            all_seeds_meta[seed_idx, tr_pct_idx] = test_performance_meta[-1][plot_evaluation_idx]
            all_seeds_itl[seed_idx, tr_pct_idx] = test_performance_itl
            all_seeds_single_task[seed_idx, tr_pct_idx] = test_performance_single_task
            all_seeds_naive[seed_idx, tr_pct_idx] = test_performance_naive

    all_subjects_naive[test_subject_idx, :] = np.nanmean(all_seeds_naive, axis=0)
    all_subjects_itl[test_subject_idx, :] = np.nanmean(all_seeds_itl, axis=0)
    all_subjects_single_task[test_subject_idx, :] = np.nanmean(all_seeds_single_task, axis=0)
    all_subjects_meta[test_subject_idx, :] = np.nanmean(all_seeds_meta, axis=0)


average_naive = np.nanmean(all_subjects_naive, axis=0)
average_itl = np.nanmean(all_subjects_itl, axis=0)
average_single_task = np.nanmean(all_subjects_single_task, axis=0)
average_meta = np.nanmean(all_subjects_meta, axis=0)

std_naive = sem(all_subjects_naive, axis=0, nan_policy='omit')
std_itl = sem(all_subjects_itl, axis=0, nan_policy='omit')
std_single_task = sem(all_subjects_single_task, axis=0, nan_policy='omit')
std_meta = sem(all_subjects_meta, axis=0, nan_policy='omit')

x_range = 100 * tr_pct_range

dpi = 100
fig, ax = plt.subplots(figsize=(1920 / dpi, 1080 / dpi), facecolor='white', dpi=dpi, nrows=1, ncols=1)
plt.plot(x_range, average_meta, 'tab:blue', linewidth=2, marker='o', linestyle='dashed')
ax.fill_between(x_range, average_meta - std_meta, average_meta + std_meta, alpha=0.1, edgecolor='tab:blue', facecolor='tab:blue', antialiased=True, label='LTL')

plt.plot(x_range, average_itl, 'tab:red', marker='o', linestyle='dotted')
ax.fill_between(x_range, average_itl - std_itl, average_itl + std_itl, alpha=0.1, edgecolor='tab:red', facecolor='tab:red', antialiased=True, label='ITL')

plt.plot(x_range, average_single_task, 'tab:green', marker='o', linestyle='dashdot')
ax.fill_between(x_range, average_single_task - std_single_task, average_single_task + std_single_task, alpha=0.1, edgecolor='tab:green', facecolor='tab:green', antialiased=True, label='Single task')

plt.plot(x_range, average_naive, 'tab:gray', marker='o', linestyle='solid')
ax.fill_between(x_range, average_naive - std_naive, average_naive + std_naive, alpha=0.1, edgecolor='tab:gray', facecolor='tab:gray', antialiased=True, label='Naive')

ax.xaxis.set_major_formatter(mtick.PercentFormatter())

# if plot_evaluation_idx == 0:
#     plt.ylim(bottom=0.7, top=2)
plt.xlabel('% meta-training')
plt.ylabel(evaluation_names[plot_evaluation_idx])
plt.legend()

figure_foldername = 'plots_' + foldername

plt.title('')
os.makedirs(figure_foldername, exist_ok=True)
plt.savefig(figure_foldername + '/vs_meta_tr_pct' + '-val_' + evaluation_names[validation_evaluation_idx] + '-plot_' + evaluation_names[plot_evaluation_idx] + '-merge_test_' + str(merge_test) + '.png', pad_inches=0)
time.sleep(0.1)

# plt.show()
k = 1
