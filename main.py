import os
import numpy as np
from src.ltl import train_test_meta
from src.independent_learning import train_test_itl
from src.naive_baseline import train_test_naive
from src.single_task import train_test_single_task
from src.utilities import save_results
from src.data_management_essex import load_data_essex_two, split_data_essex
import time
from multiprocessing import Pool



def main(settings):
    t0 = time.time()
    all_features, all_labels, all_experiment_names, all_correct = load_data_essex_two(useRT=False)
    data = split_data_essex(all_features, all_labels, all_experiment_names, settings, verbose=False, all_corr=all_correct)


    test_performance_naive, all_predictions_naive = train_test_naive(data, settings)
    test_performance_single_task, all_predictions_single_task, all_weights_single_task = train_test_single_task(data, settings)
    test_performance_itl, all_predictions_itl, all_weights_itl = train_test_itl(data, settings)
    best_model_meta, test_performance_meta, all_weight_vectors_meta, all_predictions_meta = train_test_meta(data, settings, verbose=False)

    results = {'test_performance_naive': test_performance_naive,
               'test_performance_single_task': test_performance_single_task,
               'test_performance_itl': test_performance_itl,
               'test_performance_meta': test_performance_meta,
               'all_weight_vectors_meta': all_weight_vectors_meta,
               'best_model_meta': best_model_meta,
               'all_weights_single_task': all_weights_single_task,
               'all_weights_itl': all_weights_itl,
               'settings': settings}

    save_results(results,
                 foldername='results-second_dataset_nmse_christoph_classic_range/' + 'test_subject_' + str(settings['test_subject']),
                 filename='seed_' + str(settings['seed']) + '-tr_pct_{:0.4f}'.format(settings['test_tasks_tr_points_pct']) + '-merge_test_' + str(settings['merge_test']) + '-fitness_' + settings['val_method'][0])

    #print(f'{"Naive":20s} {test_performance_naive[1]:6.4f} {test_performance_naive[-1]*100:6.4f}% \n'
    #      f'{"Single-task":20s} {test_performance_single_task[1]:6.4f} {test_performance_single_task[-1]*100:6.4f}% \n'
    #      f'{"ITL":20s} {test_performance_itl[1]:6.4f} {test_performance_itl[-1]*100:6.4f}% \n'
    #      f'{"Meta":20s} {test_performance_meta[-1][1]:6.4f} {test_performance_meta[-1][-1]*100:6.4f}%')
    print('{:0.2f}'.format(time.time() - t0) + ' s')


def parallel_calc(subj):
    test_tasks_tr_split_range = np.arange(0.0, 0.825, 0.025)
    merge_test_range = [True]
    fitness_metrics = ['NMSE']
    seed_range = np.arange(8)

    for merge_test in merge_test_range:
        for fitness in fitness_metrics:
            for curr_seed in seed_range:
                for test_tasks_tr_points_pct in test_tasks_tr_split_range:
                    print(f'test subject: {subj:2d} | merge_test: {merge_test} | fitness: {fitness:5s}'
                          f'| seed: {curr_seed:4d} | tr_pct: {test_tasks_tr_points_pct:5.3f}')
                    settings = {'regul_param_range': np.logspace(-12, 4, 100),
                               'test_subject': subj,
                               'fine_tune': True,
                               'tr_tasks_tr_points_pct': 1,
                               'val_tasks_tr_points_pct': test_tasks_tr_points_pct,
                               'test_tasks_tr_points_pct': test_tasks_tr_points_pct,
                               'evaluation': ['NMSE', 'FI', 'COR'],
                               'val_method': [fitness],
                               'merge_test': merge_test,
                               'seed': curr_seed,
                               'merge_train': False}
                    main(settings)

if __name__ == "__main__":
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    # Parameters
    with Pool(10) as p:
        p.map(parallel_calc, np.arange(10))
        p.start()
        p.join()
    print("HOLA")
