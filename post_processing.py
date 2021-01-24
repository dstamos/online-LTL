import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle

font = {'size': 48}
matplotlib.rc('font', **font)


foldername = 'results-first_test_range'
seed_range = range(1, 3)
test_tasks_tr_points_pct = 0.33 * np.linspace(0.00, 0.8, 10)[2]

all_seeds_naive = []
all_seeds_single_task = []
all_seeds_meta = []
all_seeds_itl = []
for seed in seed_range:
    filename = './' + foldername + '/' + 'seed_' + str(seed) + str(test_tasks_tr_points_pct) + '.pckl'
    try:
        test_performance_naive, test_performance_single_task, test_performance_itl, test_performance_meta, test_tasks_indexes, settings = pickle.load(open(filename, "rb"))
    except Exception as e:
        print('broken', e, seed)
        continue

    all_seeds_meta.append(test_performance_meta)
    all_seeds_itl.append(test_performance_itl)
    all_seeds_single_task.append(test_performance_single_task)
    all_seeds_naive.append(test_performance_naive)

average_meta = np.mean(all_seeds_meta, axis=0)
std_meta = np.std(all_seeds_meta, axis=0)

average_itl = np.mean(all_seeds_itl, axis=0)
std_itl = np.std(all_seeds_itl, axis=0)

average_single_task = np.mean(all_seeds_single_task, axis=0)
std_single_task = np.std(all_seeds_single_task, axis=0)

average_naive = np.mean(all_seeds_naive, axis=0)
std_naive = np.std(all_seeds_naive, axis=0)

x_range = params_to_check = range(1, len(average_meta) + 1)

dpi = 100
fig, ax = plt.subplots(figsize=(1920 / dpi, 1080 / dpi), facecolor='white', dpi=dpi, nrows=1, ncols=1)
plt.plot(x_range, average_meta, 'tab:blue')
ax.fill_between(x_range, average_meta - std_meta, average_meta + std_meta, alpha=0.1, edgecolor='tab:blue', facecolor='tab:blue', antialiased=True, label='LTL')

plt.plot(x_range, len(x_range) * [average_itl], 'tab:red')
ax.fill_between(x_range, len(x_range) * [average_itl - std_itl], len(x_range) * [average_itl + std_itl], alpha=0.1, edgecolor='tab:red', facecolor='tab:red', antialiased=True, label='ITL')

plt.plot(x_range, len(x_range) * [average_single_task], 'tab:green')
ax.fill_between(x_range, len(x_range) * [average_single_task - std_single_task], len(x_range) * [average_single_task + std_single_task], alpha=0.1, edgecolor='tab:green', facecolor='tab:green', antialiased=True, label='Single task')

plt.plot(x_range, len(x_range) * [average_naive], 'tab:gray')
ax.fill_between(x_range, len(x_range) * [average_naive - std_naive], len(x_range) * [average_naive + std_naive], alpha=0.1, edgecolor='tab:gray', facecolor='tab:gray', antialiased=True, label='Naive')

fig.tight_layout()
plt.xlabel('# training tasks')
plt.ylabel('Median Median Absolute Error')
plt.legend()
plt.savefig('errors_vs_tr_tasks_tr_9' + foldername + '.png', pad_inches=0)
plt.pause(0.1)
plt.show()
k = 1
