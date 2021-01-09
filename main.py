import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg.linalg import norm
from src.ltl import BiasLTL
from src.utilities import multiple_tasks_mae_clip
from src.data_management import concatenate_data
from sklearn.preprocessing import StandardScaler
from src.preprocessing import ThressholdScaler
from src.data_management_essex import load_data_essex, split_data_essex
from src.preprocessing import PreProcess
import pickle

matplotlib.use('Qt5Agg')


# def single_fold(data, param, each_training):
#     # Training
#     model_ltl = BiasLTL(regularization_parameter=param[0], step_size_bit=param[1])
#     outlier = ThressholdScaler()
#     sc = StandardScaler()
#     training_features, training_labels, point_indexes_per_task = concatenate_data(
#         data['training_tasks_training_features'], data['training_tasks_training_labels'])
#     training_features = outlier.fit_transform(training_features)
#     training_features = sc.fit_transform(training_features)
#     training_features = training_features / norm(training_features, axis=0, keepdims=True)
#     training_features = np.concatenate((np.ones((len(training_features), 1)), training_features), 1)
#     model_ltl.fit(training_features, training_labels, {'point_indexes_per_task': point_indexes_per_task})
#
#     # Testing. This should be kept fairly similar.
#     if len(data['validation_tasks_training_features']):
#         re_train_features, re_train_labels, re_train_indexes = concatenate_data(data['validation_tasks_training_features'],
#                                                                                 data['validation_tasks_training_labels'])
#         re_train_features = outlier.transform(re_train_features)
#         re_train_features = sc.transform(re_train_features)
#         re_train_features = re_train_features / norm(re_train_features, axis=0, keepdims=True)
#         re_train_features = np.concatenate((np.ones((len(re_train_features), 1)), re_train_features), 1)
#     else:
#         re_train_features = []
#         re_train_labels = []
#         re_train_indexes = []
#     extra_inputs = {'predictions_for_each_training_task': each_training,
#                     're_train_indexes': re_train_indexes,
#                     're_train_features': re_train_features,
#                     're_train_labels': re_train_labels
#                     }
#     test_features, test_labels, point_indexes_per_test_task = concatenate_data(
#         data['test_tasks_training_features'], data['test_tasks_training_labels'])
#     test_features = outlier.transform(test_features)
#     test_features = sc.transform(test_features)
#     test_features = test_features / norm(test_features, axis=0, keepdims=True)
#     test_features = np.concatenate((np.ones((len(test_features), 1)), test_features), 1)
#
#     extra_inputs['point_indexes_per_task'] = point_indexes_per_test_task
#     predictions_test = model_ltl.predict(test_features, extra_inputs=extra_inputs)
#     perf = multiple_tasks_mae_clip(data['test_tasks_training_labels'], predictions_test,
#                                              extra_inputs['predictions_for_each_training_task'])
#     return predictions_test, perf, (outlier, sc, model_ltl)
#
#
# def gridSearch(feats, labels, test, reg_param, ret, limit=-1):
#     nSubj = int(len(feats) / 3)
#     ind = np.arange(nSubj * 3, dtype=int)
#     test_i = np.arange(3, dtype=int) + test*3
#     bestPerf = 1
#     performance = []
#     if limit == -1 or limit >= nSubj:
#         limit = nSubj - 1
#     validate = np.arange(nSubj)
#     validate = np.setdiff1d(validate, test)
#     validate = np.random.choice(validate, limit, False)
#     param_perf = np.zeros(len(reg_param))
#     trainSkip = 3
#     if len(reg_param) > 1:
#         for i, param in enumerate(reg_param):#Search for each possible regurlaization paramether
#             for s in validate: #In a cross validation manner
#                 val = np.arange(3, dtype=int) + s*3
#                 train = np.setdiff1d(ind, test_i)
#                 train = np.setdiff1d(train, val)
#                 data = split_data_essex(feats, labels, train, val[:ret], val[ret:])
#                 foo, perf, foo = single_fold(data, param, False)
#                 performance.append(perf)
#             param_perf[i] = np.mean(performance)
#             if param_perf[i] < bestPerf: #Save the best one
#                 bestPerf = param_perf[i]
#                 best_param = param
#     else:
#         best_param = reg_param[0]
#     train = np.setdiff1d(ind, test_i)
#     data = split_data_essex(feats, labels, np.random.permutation(train), test_i[:ret], test_i[ret:])  # Retrain with train + val with the best
#     prediction, perf, mdl = single_fold(data, best_param, True)
#     return param_perf, prediction, perf, mdl
#
#
# def calculateLastDay(labels, prediction):
#     res = np.zeros((3, 8, 21))
#     for ap in range(3):
#         for s in range(8):
#             y = labels[s*3 + 2]
#             for d in range(21):
#                 p = prediction[ap*8 + s][d][-1]
#                 res[ap, s, d] =ThressholdScaler np.median(np.abs(y - np.clip(p, .1, 1)))
#     return res


def main(settings, seed):
    np.random.seed(seed)

    # Load and split data datasets
    all_features, all_labels, all_experiment_names = load_data_essex(useRT=False)
    data = split_data_essex(all_features, all_labels, all_experiment_names, settings)

    # Preprocess the data
    pre = PreProcess(threshold_scaling=True, standard_scaling=True, inside_ball_scaling=True, add_bias=True)
    tr_tasks_tr_features, tr_tasks_tr_labels = pre.transform(data['tr_tasks_tr_features'], data['tr_tasks_tr_labels'], training=True)

    # Training
    for regul_param in settings['regul_param_range']:
        # Optimise metaparameters on the training tasks.
        model_ltl = BiasLTL(regul_param=regul_param, step_size_bit=1, keep_all_metaparameters=True)
        model_ltl.fit_meta(tr_tasks_tr_features, tr_tasks_tr_labels)

        # Fine-tune on the validation tasks.
        val_tasks_tr_features, val_tasks_tr_labels, val_tr_point_indexes_per_task = pre.transform(data['val_tasks_tr_features'], data['val_tasks_tr_labels'])
        weight_vectors = model_ltl.fit_inner(val_tasks_tr_features, val_tasks_tr_labels, {'point_indexes_per_task': val_tr_point_indexes_per_task})

        # model_ltl.predict(val_tasks_tr_features, extra_inputs={'point_indexes_per_task': val_tr_point_indexes_per_task, 'predictions_for_each_training_task': False})

        # Check performance on the validation tasks.
    # Test
    pass

    # Plot
    pass


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
    seed_range = range(1, 31)
    regul_param_range = np.logspace(-6, 4, 36)
    # TODO Is this needed?
    iteratations_over_each_task = 3

    n_test_subjects = 1
    fine_tuning = True  # Fine-tuning is the process of customizing the metalearning model on the test tasks. That typically includes re-training on a small number of datapoints.
    # TODO Will probably need settings for fine-tuning on day 0 of the test subject and straight up testing on days 1 and 2

    # Dataset split for training tasks (only training points)
    tr_tasks_tr_points_pct = 0.2

    # Dataset split for validation tasks (only training+validation points)
    val_tasks_tr_points_pct = 0.2
    val_tasks_val_points_pct = 0.3

    # Dataset split for test tasks
    test_tasks_tr_points_pct = 0.2
    test_tasks_val_points_pct = 0.3
    test_tasks_test_points_pct = 0.5
    assert test_tasks_tr_points_pct + test_tasks_val_points_pct + test_tasks_test_points_pct == 1, 'Percentages need to add up to 1'

    options = {'regul_param_range': regul_param_range,
               'iteratations_over_each_task': iteratations_over_each_task,
               'n_test_subjects': n_test_subjects,
               'fine_tuning': fine_tuning,
               'tr_tasks_tr_points_pct': tr_tasks_tr_points_pct,
               'val_tasks_tr_points_pct': val_tasks_tr_points_pct,
               'val_tasks_val_points_pct': val_tasks_val_points_pct,
               'test_tasks_tr_points_pct': test_tasks_tr_points_pct,
               'test_tasks_val_points_pct': test_tasks_val_points_pct,
               'test_tasks_test_points_pct': test_tasks_test_points_pct}

    for curr_seed in seed_range:
        main(options, curr_seed)

    # all_features, all_labels = load_data_essex(useRT=False)
    #
    # nSubj = int(len(all_features) / 3)
    # foldLimit = 3  # nSubj- 1
    # prediction = []
    # models = []
    # bestmp = []
    # performance = np.zeros((nSubj, (nSubj - 1) * 3))
    #
    # # reg_param = np.logspace(-6, 3, 10)
    # reg_param = np.logspace(-5, 1, 7)
    # step_size = [1]
    # params = [(x, y) for x in reg_param for y in step_size]
    # for retrain in range(3):  # 3
    #     for s in range(nSubj):  # Leave one subject out retrain 1 day
    #         pp, pred, perf, mdl = gridSearch(all_features, all_labels, s, params, retrain, foldLimit)
    #         performance[s] = perf
    #         prediction.append(pred)
    #         models.append(mdl)
    #         bestmp.append(pp)
    #     np.save('performance_' + str(retrain) + '_retrain_doubleNorm.npy', performance)
    #     with open('results_' + str(retrain) + '_retrain_doubleNorm.pkl', 'wb') as f:
    #         pickle.dump([prediction, models, bestmp], f)
    #     plt.plot(np.mean(performance, 0))
    #
    # if False:
    #     # Independent learning on the test tasks.
    #     from src.independent_learning import ITL
    #     test_tasks_training_features, test_tasks_training_labels, point_indexes_per_test_task = concatenate_data(data['test_tasks_training_features'], data['test_tasks_training_labels'])
    #     extra_inputs = {'point_indexes_per_task': point_indexes_per_test_task}
    #
    #     model_itl = ITL(regularization_parameter=1e-6)
    #     model_itl.fit(test_tasks_training_features, test_tasks_training_labels, extra_inputs=extra_inputs)
    #
    #     test_tasks_validation_features, _, point_indexes_per_test_task = concatenate_data(data['test_tasks_validation_features'], data['test_tasks_validation_labels'])
    #     extra_inputs = {'point_indexes_per_task': point_indexes_per_test_task}
    #     predictions_validation = model_itl.predict(test_tasks_validation_features, extra_inputs=extra_inputs)
    #
    #     val_performance = multiple_tasks_mse(data['test_tasks_validation_labels'], predictions_validation)
    #     print(val_performance)
    #
    #     test_tasks_test_features, _, point_indexes_per_test_task = concatenate_data(data['test_tasks_test_features'], data['test_tasks_test_labels'])
    #     extra_inputs = {'point_indexes_per_task': point_indexes_per_test_task}
    #     predictions_test = model_itl.predict(test_tasks_test_features, extra_inputs=extra_inputs)
    #
    #     itl_test_performance = multiple_tasks_mse(data['test_tasks_test_labels'], predictions_test)
    #     print(itl_test_performance)
    #     ###################################################################################################
    #     ###################################################################################################
    #     ###################################################################################################
    #
    #     import matplotlib.pyplot as plt
    #     plt.plot(ltl_test_performance, color='tab:red', label='BiasLTL')
    #     plt.axhline(y=itl_test_performance, xmin=0, xmax=len(ltl_test_performance)-1, color='tab:blue', label='ITL')
    #     plt.show()
    #
    #     print('done')
