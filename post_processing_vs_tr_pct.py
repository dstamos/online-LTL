import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import os
import time
import matplotlib.ticker as mtick

font = {'size': 36}
matplotlib.rc('font', **font)

seed_range = range(30)
test_subject_range = range(8)
merge_test = False
evaluation_idx = 1

evaluation_names = ['MAE', 'NMSE', 'MSE', 'MCA', 'CD']

all_errors_itl = []
all_errors_naive = []
for test_subject in test_subject_range:
    foldername = 'results-first_dataset_nmse_30_seeds'
    foldername_with_subfolder = foldername + '/test_subject_' + str(test_subject)
    tr_val_pct_range = np.arange(0.0, 0.625, 0.025)
    test_tasks_tr_points_pct = tr_val_pct_range

    all_seeds_naive = np.full([len(seed_range), len(test_tasks_tr_points_pct)], np.nan)
    all_seeds_single_task = np.full([len(seed_range), len(test_tasks_tr_points_pct)], np.nan)
    all_seeds_meta = np.full([len(seed_range), len(test_tasks_tr_points_pct)], np.nan)
    all_seeds_itl = np.full([len(seed_range), len(test_tasks_tr_points_pct)], np.nan)
    for tr_pct_idx, tr_pct in enumerate(test_tasks_tr_points_pct):
        for seed_idx, seed in enumerate(seed_range):
            filename = './' + foldername_with_subfolder + '/' + 'seed_' + str(seed) + '-tr_pct_{:0.4f}'.format(tr_pct)+'-merge_test_'+str(merge_test) +  '-fitness_' + evaluation_names[evaluation_idx] + '.pckl'
            # filename = './' + foldername + '/' + 'seed_' + str(seed) + '-tr_pct_' + str(tr_pct) + '.pckl'
            try:
                results = pickle.load(open(filename, "rb"))
                test_performance_naive = results['test_performance_naive'][evaluation_idx]
                test_performance_single_task = results['test_performance_single_task'][evaluation_idx]
                test_performance_itl = results['test_performance_itl'][evaluation_idx]
                test_performance_meta = results['test_performance_meta']
                all_weight_vectors_meta = results['all_weight_vectors_meta']
                best_model_meta = results['best_model_meta']
                settings = results['settings']

            except Exception as e:
                print('broken', test_subject, tr_pct_idx, seed)
                continue

            all_seeds_meta[seed_idx, tr_pct_idx] = test_performance_meta[-1][evaluation_idx]
            all_seeds_itl[seed_idx, tr_pct_idx] = test_performance_itl
            all_seeds_single_task[seed_idx, tr_pct_idx] = test_performance_single_task
            all_seeds_naive[seed_idx, tr_pct_idx] = test_performance_naive

    average_meta = np.nanmean(all_seeds_meta, axis=0)
    std_meta = np.nanstd(all_seeds_meta, axis=0)

    average_itl = np.nanmean(all_seeds_itl, axis=0)
    std_itl = np.nanstd(all_seeds_itl, axis=0)

    average_single_task = np.nanmean(all_seeds_single_task, axis=0)
    std_single_task = np.nanstd(all_seeds_single_task, axis=0)

    average_naive = np.nanmean(all_seeds_naive, axis=0)
    std_naive = np.nanstd(all_seeds_naive, axis=0)

    all_errors_itl.append(average_itl)
    all_errors_naive.append(average_naive)
    x_range = 100 * tr_val_pct_range

    dpi = 100
    fig, ax = plt.subplots(figsize=(1920 / dpi, 1080 / dpi), facecolor='white', dpi=dpi, nrows=1, ncols=1)
    plt.plot(x_range, average_meta, 'tab:blue', linewidth=2, linestyle='-', marker='o')
    ax.fill_between(x_range, average_meta - std_meta, average_meta + std_meta, alpha=0.1, edgecolor='tab:blue', facecolor='tab:blue', antialiased=True, label='LTL')

    plt.plot(x_range, average_itl, 'tab:red', marker='o')
    ax.fill_between(x_range, average_itl - std_itl, average_itl + std_itl, alpha=0.1, edgecolor='tab:red', facecolor='tab:red', antialiased=True, label='ITL')

    plt.plot(x_range, average_single_task, 'tab:green', marker='o')
    ax.fill_between(x_range, average_single_task - std_single_task, average_single_task + std_single_task, alpha=0.1, edgecolor='tab:green', facecolor='tab:green', antialiased=True, label='Single task')

    plt.plot(x_range, average_naive, 'tab:gray', marker='o')
    ax.fill_between(x_range, average_naive - std_naive, average_naive + std_naive, alpha=0.1, edgecolor='tab:gray', facecolor='tab:gray', antialiased=True, label='Naive')

    ax.xaxis.set_major_formatter(mtick.PercentFormatter())

    if evaluation_idx == 1:
        plt.ylim(top=2)
    plt.title('subject ' + str(test_subject))
    plt.xlabel('training %')
    plt.ylabel('performance')
    plt.legend()

    # figure_foldername = 'plots_' + foldername
    # os.makedirs(figure_foldername, exist_ok=True)
    # plt.savefig(figure_foldername + '/' + evaluation_names[evaluation_idx] + '_test_subject_' + str(test_subject) + '-merge_test_' + str(merge_test) + '.png', pad_inches=0)
    # time.sleep(0.1)

# plt.show()
k = 1
