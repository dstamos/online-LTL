import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
from src.ltl import train_test_meta
from src.independent_learning import train_test_itl
from src.naive_baseline import train_test_naive
from src.naive_transfer_baseline import train_test_naive_transfer
from src.single_task import train_test_single_task
from src.utilities import save_results
from src.data_management_essex import load_data_essex_one, load_data_essex_two, split_data_essex
import pickle
import sys

# matplotlib.use('Qt5Agg')


def main(settings, seed):
    np.random.seed(seed)

    all_features, all_labels, all_experiment_names = load_data_essex_two(useRT=False)
    data = split_data_essex(all_features, all_labels, all_experiment_names, settings, verbose=True)

    test_performance_naive = train_test_naive(data, settings)

    # test_performance_naive_transfer = train_test_naive_transfer(data, settings) # TODO
    test_performance_naive_transfer = np.nan

    # test_performance_single_task = train_test_single_task(data, settings)
    test_performance_single_task = np.nan

    test_performance_itl = train_test_itl(data, settings)

    # best_model_meta, test_performance_meta = train_test_meta(data, settings, verbose=False)
    test_performance_meta = [np.nan]    # Because this output is a list by default

    results = {'test_performance_naive': test_performance_naive,
               'test_performance_naive_transfer': test_performance_naive_transfer,
               'test_performance_single_task': test_performance_single_task,
               'test_performance_itl': test_performance_itl,
               'test_performance_meta': test_performance_meta,
               'settings': settings}

    save_results(results,
                 foldername='results-second_dataset/' + 'test_subject_' + str(settings['test_subject']),
                 filename='seed_' + str(seed) + '-tr_pct_' + str(settings['test_tasks_tr_points_pct']))

    print(f'{"Naive":20s} {test_performance_naive:6.4f} \n'
          f'{"Naive (transfer)":20s} {test_performance_naive_transfer:6.4f} \n'
          f'{"Single-task":20s} {test_performance_single_task:6.4f} \n'
          f'{"ITL":20s} {test_performance_itl:6.4f} \n'
          f'{"Meta":20s} {test_performance_meta[-1]:6.4f}')


if __name__ == "__main__":

    """
    The BiasLTL metalearning pipeline:
    a) Take your T tasks. Split them into training/validation/test tasks.
    b) You train the "centroid"/metaparameter on the training tasks.
    c) You go to the validation tasks, fine-tune the model on each task (on training points) and check the performance (on test points).
    d) Pick the metaparameter that resulted in the best average performance on the validation tasks.
    e) Go to the test tasks using the optimal metaparameter, fine-tune on a small number of points (or don't) and test the performance.
    """

    # Parameters
    # test_subject_range = range(0, 10)
    test_subject_range = [0]

    # test_tasks_tr_split_range = np.linspace(0.00, 0.8, 30)
    test_tasks_tr_split_range = np.array([0.2])

    if len(sys.argv) > 1:
        # This is the case when main.py is called from a bash script with inputs
        seed_range = [int(sys.argv[1])]
        test_tasks_tr_split_range = np.array([test_tasks_tr_split_range[int(sys.argv[2])]])
    else:
        seed_range = [0]
    regul_param_range = np.logspace(-16, 5, 64)

    fine_tune = True  # Fine-tuning is the process of customizing the metalearning model on the test tasks. That typically includes re-training on a small number of datapoints.

    # Dataset split for training tasks (only training points)
    tr_tasks_tr_points_pct = 0.2

    val_tasks_tr_points_pct = 0.5
    val_tasks_test_points_pct = 0.5
    assert val_tasks_tr_points_pct + val_tasks_test_points_pct == 1, 'Percentages need to add up to 1'

    test_tasks_tr_points_pct_range = test_tasks_tr_split_range
    test_tasks_test_points_pct_range = 1 - test_tasks_tr_points_pct_range
    assert np.all(test_tasks_tr_points_pct_range + test_tasks_test_points_pct_range == 1), 'Percentages need to add up to 1'

    for curr_test_subject in test_subject_range:
        for curr_seed in seed_range:
            for test_tasks_tr_points_pct, test_tasks_test_points_pct in zip(test_tasks_tr_points_pct_range, test_tasks_test_points_pct_range):
                print(f'test subject: {curr_test_subject:2d} | seed: {curr_seed:2d} | tr_pct: {test_tasks_tr_points_pct:5.3f}')
                options = {'regul_param_range': regul_param_range,
                           'test_subject': curr_test_subject,
                           'fine_tune': fine_tune,
                           'tr_tasks_tr_points_pct': tr_tasks_tr_points_pct,
                           'val_tasks_tr_points_pct': val_tasks_tr_points_pct,
                           'val_tasks_test_points_pct': val_tasks_test_points_pct,
                           'test_tasks_tr_points_pct': test_tasks_tr_points_pct,
                           'test_tasks_test_points_pct': test_tasks_test_points_pct}
                main(options, curr_seed)
                print('\n')
