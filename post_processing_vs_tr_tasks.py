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
# test_subject_range = [9]
merge_test = False
validation_evaluation_idx = 0
plot_evaluation_idx = 2
tr_pct = 0.25

evaluation_names = ['NMSE', 'FI', 'COR']
foldername = 'results'

all_subjects_naive = []
all_subjects_single_task = []
all_subjects_meta = []
all_subjects_itl = []

for test_subject_idx, test_subject in enumerate(test_subject_range):
    all_seeds_naive = []
    all_seeds_single_task = []
    all_seeds_meta = []
    all_seeds_itl = []
    for seed_idx, seed in enumerate(seed_range):
        foldername_with_subfolder = foldername + '/test_subject_' + str(test_subject)

        filename = './' + foldername_with_subfolder + '/' + 'seed_' + str(seed) + '-tr_pct_{:0.4f}'.format(tr_pct)+'-merge_test_'+str(merge_test) + '-fitness_' + evaluation_names[validation_evaluation_idx] + '.pckl'
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

        all_seeds_meta.append([test_performance_meta[i][plot_evaluation_idx] for i in range(len(test_performance_meta))])
        all_seeds_itl.append(test_performance_itl)
        all_seeds_single_task.append(test_performance_single_task)
        all_seeds_naive.append(test_performance_naive)

    all_seeds_meta = np.nanmean(all_seeds_meta, axis=0)
    all_seeds_itl = np.nanmean(all_seeds_itl, axis=0)
    all_seeds_single_task = np.nanmean(all_seeds_single_task, axis=0)
    all_seeds_naive = np.nanmean(all_seeds_naive, axis=0)

    try:
        all_subjects_naive.append(all_seeds_naive)
        all_subjects_itl.append(all_seeds_itl)
        all_subjects_single_task.append(all_seeds_single_task)
        all_subjects_meta.append(all_seeds_meta)
    except:
        continue

average_naive = np.nanmean(all_subjects_naive, axis=0)
average_itl = np.nanmean(all_subjects_itl, axis=0)
average_single_task = np.nanmean(all_subjects_single_task, axis=0)
average_meta = np.nanmean(all_subjects_meta, axis=0)

std_naive = sem(all_subjects_naive, axis=0, nan_policy='omit')
std_itl = sem(all_subjects_itl, axis=0, nan_policy='omit')
std_single_task = sem(all_subjects_single_task, axis=0, nan_policy='omit')
std_meta = sem(all_subjects_meta, axis=0, nan_policy='omit')

x_range = params_to_check = range(1, len(average_meta) + 1)

average_naive = np.array(len(average_meta) * [average_naive])
average_itl = np.array(len(average_meta) * [average_itl])
average_single_task = np.array(len(average_meta) * [average_single_task])

std_naive = np.array(len(average_meta) * [std_naive])
std_itl = np.array(len(average_meta) * [std_itl])
std_single_task = np.array(len(average_meta) * [std_single_task])


dpi = 100
fig, ax = plt.subplots(figsize=(1920 / dpi, 1080 / dpi), facecolor='white', dpi=dpi, nrows=1, ncols=1)
ax.plot(x_range, average_meta, 'tab:blue', linewidth=2, marker='o', linestyle='dashed')
ax.fill_between(x_range, average_meta - std_meta, average_meta + std_meta, alpha=0.1, edgecolor='tab:blue', facecolor='tab:blue', antialiased=True, label='LTL')

ax.plot(x_range, average_itl, 'tab:red', marker='o', linestyle='dotted')
ax.fill_between(x_range, average_itl - std_itl, average_itl + std_itl, alpha=0.1, edgecolor='tab:red', facecolor='tab:red', antialiased=True, label='ITL')

ax.plot(x_range, average_single_task, 'tab:green', marker='o', linestyle='dashdot')
ax.fill_between(x_range, average_single_task - std_single_task, average_single_task + std_single_task, alpha=0.1, edgecolor='tab:green', facecolor='tab:green', antialiased=True, label='Single task')

# ax.plot(x_range, average_naive, 'tab:gray', marker='o', linestyle='solid')
# ax.fill_between(x_range, average_naive - std_naive, average_naive + std_naive, alpha=0.1, edgecolor='tab:gray', facecolor='tab:gray', antialiased=True, label='Naive')

plt.xlabel('# training tasks')
plt.ylabel(evaluation_names[plot_evaluation_idx])
plt.legend()

figure_foldername = 'plots_' + foldername

plt.title('')
os.makedirs(figure_foldername, exist_ok=True)
plt.savefig(figure_foldername + '/vs_tr_tasks' + '-val_' + evaluation_names[validation_evaluation_idx] + '-plot_' + evaluation_names[plot_evaluation_idx] + '-merge_test_' + str(merge_test) + '-tr_pct_' + str(tr_pct) + '.png', pad_inches=0)
time.sleep(0.1)

# plt.show()
k = 1
