import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle

font = {'size': 48}
matplotlib.rc('font', **font)


seed_range = range(1, 6)

all_seeds_meta = []
all_seeds_itl = []
for seed in seed_range:
    filename = './results/' + 'seed_' + str(seed) + '.pckl'
    try:
        test_performance_itl, test_performance_meta, test_tasks_indexes, settings = pickle.load(open(filename, "rb"))
    except Exception as e:
        print('broken', e, seed)
        continue

    all_seeds_meta.append(test_performance_meta)
    all_seeds_itl.append(test_performance_itl)

average_meta = np.mean(all_seeds_meta, axis=0)
std_meta = np.std(all_seeds_meta, axis=0)

average_itl = np.mean(all_seeds_itl, axis=0)
std_itl = np.std(all_seeds_itl, axis=0)

x_range = params_to_check = range(1, len(average_meta) + 1)

dpi = 100
fig, ax = plt.subplots(figsize=(1920 / dpi, 1080 / dpi), facecolor='white', dpi=dpi, nrows=1, ncols=1)
plt.plot(x_range, average_meta, 'tab:blue')
ax.fill_between(x_range, average_meta - std_meta, average_meta + std_meta, alpha=0.1, edgecolor='tab:blue', facecolor='tab:blue', antialiased=True, label='LTL')

plt.plot(x_range, len(x_range) * [average_itl], 'tab:gray')
ax.fill_between(x_range, len(x_range) * [average_itl - std_itl], len(x_range) * [average_itl + std_itl], alpha=0.1, edgecolor='tab:gray', facecolor='tab:gray', antialiased=True, label='ITL')

fig.tight_layout()
plt.xlabel('# training tasks')
plt.ylabel('Median Median Absolute Error')
plt.legend()
plt.savefig('errors_vs_tr_tasks.png', pad_inches=0)
plt.pause(0.1)
k = 1
