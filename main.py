import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
from src.ltl import train_test_meta
from src.independent_learning import train_test_itl
from src.naive_baseline import train_test_naive
from src.single_task import train_test_single_task
from src.data_management_essex import load_data_essex_one, load_data_essex_two, split_data_essex
import pickle
import sys

# matplotlib.use('Qt5Agg')


def main(settings, seed):
    np.random.seed(seed)

    # Load and split data datasets
    all_features, all_labels, all_experiment_names = load_data_essex_two(useRT=False)
    data = split_data_essex(all_features, all_labels, all_experiment_names, settings, verbose=False)

    test_performance_naive = train_test_naive(data, settings)
    test_performance_single_task = train_test_single_task(data, settings)
    test_performance_itl = train_test_itl(data, settings)
    best_model_meta, test_performance_meta = train_test_meta(data, settings, verbose=False)

    foldername = 'results-second_dataset/' + 'test_subject_' + str(settings['test_subject'])
    os.makedirs(foldername, exist_ok=True)
    filename = './' + foldername + '/' + 'seed_' + str(seed) + '-tr_pct_' + str(settings['test_tasks_tr_points_pct']) + '.pckl'

    pickle.dump([test_performance_naive, test_performance_single_task, test_performance_itl, test_performance_meta, data['test_tasks_indexes'], settings], open(filename, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    # font = {'size': 24}
    # matplotlib.rc('font', **font)
    # my_dpi = 100
    # fig, ax = plt.subplots(figsize=(1920 / my_dpi, 1080 / my_dpi), facecolor='white', dpi=my_dpi, nrows=1, ncols=1)
    # ax.plot(range(1, len(data['training_tasks_indexes']) + 1), [test_performance_itl] * len(data['training_tasks_indexes']), linewidth=2, color='tab:red', label='Independent Learning')
    # ax.plot(range(1, len(data['training_tasks_indexes']) + 1), [test_performance_naive] * len(data['training_tasks_indexes']), linewidth=2, color='tab:gray', label='Naive Baseline')
    # ax.plot(range(1, len(data['training_tasks_indexes']) + 1), [test_performance_single_task] * len(data['training_tasks_indexes']), linewidth=2, color='black', label='Single task')
    # ax.plot(range(1, len(test_performance_meta) + 1), test_performance_meta, linewidth=2, color='tab:blue', label='Bias Meta-learning')
    #
    # plt.xlabel('# training tasks')
    # plt.ylabel('test performance')
    # plt.legend()
    # plt.savefig('cv' + '.png', pad_inches=0)
    # plt.pause(0.1)
    # # exit()
    # plt.show()


if __name__ == "__main__":

    """
    The BiasLTL metalearning pipeline:
    a) Take your T tasks. Split them into training/validation/test tasks.
    b) You train the "centroid"/metaparameter on the training tasks.
    c) You go to the validation tasks, fine-tune the model on each task (on training points) and check the performance (on validation points).
    d) Pick the metaparameter that resulted in the best average performance on the validation tasks.
    e) Go to the test tasks using the optimal metaparameter, fine-tune on a small number of points (or don't) and test the performance.
    """

    # Parameters
    test_subject_range = range(0, 10)
    test_tasks_tr_split_range = np.linspace(0.00, 0.8, 30)
    # test_tasks_tr_split_range = np.array([0.5])

    if len(sys.argv) > 1:
        seed_range = [int(sys.argv[1])]
        test_tasks_tr_split_range = np.array([test_tasks_tr_split_range[int(sys.argv[2])]])
        # test_subject_range = [test_subject_range[int(sys.argv[3)]]
    else:
        seed_range = range(1, 11)
    regul_param_range = np.logspace(-10, 4, 49)

    fine_tune = True  # Fine-tuning is the process of customizing the metalearning model on the test tasks. That typically includes re-training on a small number of datapoints.
    # TODO Will probably need settings for fine-tuning on day 0 of the test subject and straight up testing on days 1 and 2

    # Dataset split for training tasks (only training points)
    tr_tasks_tr_points_pct = 0.2

    # Dataset split for validation tasks (only training+validation points)
    val_tasks_tr_points_pct = 0.2
    val_tasks_val_points_pct = 0.3
    val_tasks_test_points_pct = 0.5
    assert val_tasks_tr_points_pct + val_tasks_val_points_pct + val_tasks_test_points_pct == 1, 'Percentages need to add up to 1'

    # Dataset split for test tasks
    test_tasks_tr_points_pct_range = 0.5 * test_tasks_tr_split_range
    test_tasks_val_points_pct_range = (1 - 0.5) * test_tasks_tr_split_range
    test_tasks_test_points_pct_range = 1 - (test_tasks_tr_points_pct_range + test_tasks_val_points_pct_range)
    assert np.all(test_tasks_tr_points_pct_range + test_tasks_val_points_pct_range + test_tasks_test_points_pct_range == 1), 'Percentages need to add up to 1'

    for curr_test_subject in test_subject_range:
        for curr_seed in seed_range:
            for test_tasks_tr_points_pct, test_tasks_val_points_pct, test_tasks_test_points_pct in zip(test_tasks_tr_points_pct_range, test_tasks_val_points_pct_range, test_tasks_test_points_pct_range):
                print(f'test subject: {curr_test_subject:2d} | seed: {curr_seed:2d} | tr_pct: {test_tasks_tr_points_pct:5.3f}')
                options = {'regul_param_range': regul_param_range,
                           'test_subject': curr_test_subject,
                           'fine_tune': fine_tune,
                           'tr_tasks_tr_points_pct': tr_tasks_tr_points_pct,
                           'val_tasks_tr_points_pct': val_tasks_tr_points_pct,
                           'val_tasks_val_points_pct': val_tasks_val_points_pct,
                           'val_tasks_test_points_pct': val_tasks_test_points_pct,
                           'test_tasks_tr_points_pct': test_tasks_tr_points_pct,
                           'test_tasks_val_points_pct': test_tasks_val_points_pct,
                           'test_tasks_test_points_pct': test_tasks_test_points_pct}
                main(options, curr_seed)
                print('\n')
