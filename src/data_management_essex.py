import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR
from src.utilities import evaluation_methods


def load_data_essex_one(delete0=True, useStim=True, useRT=True, exclude=[7]):
    extra = np.load('./data/extra.npy')
    correct = np.load('./data/corr.npy')
    if useStim:
        stim = np.load('./data/stimFeatures.npy')
    resp = np.load('./data/respFeatures.npy')
    feat = []
    label = []
    corr = []
    for s in np.unique(extra[:, 0]):
        if s not in exclude:
            for d in range(1, 4):
                val = np.logical_and(extra[:, 0] == s, extra[:, 1] == d)
                if delete0:
                    val = np.logical_and(val, extra[:, 3] != 0)
                nval = np.sum(val)
                f = resp[val, :]
                if useStim:
                    f = np.concatenate((f, stim[val, :]), 1)
                if useRT:
                    f = np.concatenate((f, np.expand_dims(extra[val, 2], 1)), 1)
                feat.append(f)
                label.append(extra[val, 3])
                corr.append(correct[val])

    # The assumption is that each subject had 3 days of experiments.
    # The point of this is to make it easy to check for mistakes down the line
    n_subjects = len(feat) // 3
    experiment_names = []
    for curr_subject in range(n_subjects):
        day = 0
        while day < 3:
            task_name = 'subject_' + str(curr_subject) + '-day_' + str(day)
            experiment_names.append(task_name)
            day = day + 1

    return feat, label, experiment_names, corr


def load_data_essex_two(useRT=True):

    l = np.load('./data/confidence_Chris.npy', allow_pickle=True)
    f = np.load('./data/features_Chris.npy', allow_pickle=True)
    c = np.load('./data/correctness_Chris.npy', allow_pickle=True)
    features = [i for i in f]

    labels = [i for i in l]
    corr = [i.astype(bool) for i in c]

    if not(useRT):
        features = [f[:, :-1] for f in features]

    # The assumption is that each subject had 3 days of experiments.
    # The point of this is to make it easy to check for mistakes down the line
    n_subjects = len(features) // 3
    experiment_names = []
    for curr_subject in range(n_subjects):
        day = 0
        while day < 3:
            task_name = 'subject_' + str(curr_subject) + '-day_' + str(day)
            experiment_names.append(task_name)
            day = day + 1

    return features, labels, experiment_names, corr


def split_data_essex(all_features, all_labels, all_experiment_names, settings, verbose=True, all_corr=[]):
    """
    Training tasks only have training data.
    Validation tasks only have training and test data.
    Test tasks have training, validation and test data. The validation data are currently not used for metalearning, only for independent learning.
    :param all_experiment_names: list of experiment names
    :param all_features: list of numpy arrays (n_points, dims)
    :param all_labels:  list of numpy arrays (n_points, )
    :param settings:  dict of settings
    :param all_corr: List of numpy arrays (n_points, )
    :return:
    """

    """
    This function is very much hardcoded for the setting in which we have T subjects with 3 experiments per subject (3 * T total experiments/tasks). 
    We leave n_test_subjects subjects out for testing (3 experiments for each), and 3 experiments out for validation (they can belong to ANY subject).
    """
    # FIXME A bunch of hardcoded things in this function.
    # Pick the test_subjects, find the corresponding test_tasks_indexes.
    n_experiments_per_subject = 3
    n_all_subjects = len(all_features) // n_experiments_per_subject  # Hardcoded - the assumption is that all subjects had 3 days of experiments
    test_subjects = [settings['test_subject']]

    days = settings['test_tasks_tr_points_pct']

    test_tasks_indexes = []
    for test_subject in test_subjects:
        curr_indexes = test_subject * n_experiments_per_subject + np.arange(0, 3)
        test_tasks_indexes = test_tasks_indexes + curr_indexes.tolist()

    tasks_indexes = list(range(0, n_all_subjects * n_experiments_per_subject))
    for idx in test_tasks_indexes:
        tasks_indexes.remove(idx)

    # Test tasks (training and test data)
    test_tasks_tr_features = []
    test_tasks_tr_labels = []
    test_tasks_tr_corr = []
    test_tasks_test_features = []
    test_tasks_test_labels = []
    test_tasks_test_corr = []

    if days == 1:
        training_features = all_features[test_tasks_indexes[0]]
        training_labels = all_labels[test_tasks_indexes[0]]
        training_corr = all_corr[test_tasks_indexes[0]]
    else:
        training_features = np.concatenate((all_features[test_tasks_indexes[0]], all_features[test_tasks_indexes[0]]), axis=0)
        training_labels = np.concatenate((all_labels[test_tasks_indexes[0]], all_labels[test_tasks_indexes[0]]), axis=0)
        training_corr = np.concatenate((all_corr[test_tasks_indexes[0]], all_corr[test_tasks_indexes[0]]), axis=0)
    test_features = all_features[test_tasks_indexes[-1]]
    test_labels = all_labels[test_tasks_indexes[-1]]
    test_corr = all_corr[test_tasks_indexes[-1]]
    test_tasks_indexes = [0]

    test_tasks_tr_features.append(training_features)
    test_tasks_tr_labels.append(training_labels)
    test_tasks_tr_corr.append(training_corr)
    test_tasks_test_features.append(test_features)
    test_tasks_test_labels.append(test_labels)
    test_tasks_test_corr.append(test_corr)

    # Validation tasks are picked randomly (not from the same person)

    aval_subj = np.unique(np.array(tasks_indexes) // 3)
    validation_tasks_indexes = aval_subj[settings['seed']] * n_experiments_per_subject + np.arange(0, 3)
    tasks_indexes = [i for i in tasks_indexes if i not in validation_tasks_indexes]
    training_tasks_indexes = tasks_indexes

    # Training tasks (only training data)
    tr_tasks_tr_features = []
    tr_tasks_tr_labels = []
    tr_tasks_tr_corr = []
    for counter, task_index in enumerate(training_tasks_indexes):
        tr_tasks_tr_features.append(all_features[task_index])
        tr_tasks_tr_labels.append(all_labels[task_index])
        tr_tasks_tr_corr.append(all_corr[task_index])

        if verbose is True:
            print(f'task: {all_experiment_names[task_index]:s} ({task_index:2d}) | points: {n_all_points:4d} | tr: {n_tr_points:4d}')

    # Validation tasks (training and test data)
    val_tasks_tr_features = []
    val_tasks_tr_labels = []
    val_tasks_tr_corr = []
    val_tasks_test_features = []
    val_tasks_test_labels = []
    val_tasks_test_corr = []

    if days == 1:
        training_features = all_features[validation_tasks_indexes[0]]
        training_labels = all_labels[validation_tasks_indexes[0]]
        training_corr = all_corr[validation_tasks_indexes[0]]
    else:
        training_features = np.concatenate((all_features[validation_tasks_indexes[0]], all_features[validation_tasks_indexes[0]]), axis=0)
        training_labels = np.concatenate((all_labels[validation_tasks_indexes[0]], all_labels[validation_tasks_indexes[0]]), axis=0)
        training_corr = np.concatenate((all_corr[validation_tasks_indexes[0]], all_corr[validation_tasks_indexes[0]]), axis=0)
    test_features = all_features[validation_tasks_indexes[-1]]
    test_labels = all_labels[validation_tasks_indexes[-1]]
    test_corr = all_corr[validation_tasks_indexes[-1]]
    validation_tasks_indexes = [0]

    val_tasks_tr_features.append(training_features)
    val_tasks_tr_labels.append(training_labels)
    val_tasks_tr_corr.append(training_corr)
    val_tasks_test_features.append(test_features)
    val_tasks_test_labels.append(test_labels)
    val_tasks_test_corr.append(test_corr)

    data = {'training_tasks_indexes': training_tasks_indexes,
        'validation_tasks_indexes': validation_tasks_indexes,
        'test_tasks_indexes': test_tasks_indexes,
        # Training tasks
        'tr_tasks_tr_features': tr_tasks_tr_features,
        'tr_tasks_tr_labels': tr_tasks_tr_labels,
        'tr_tasks_tr_corr': tr_tasks_tr_corr,
        # Validation tasks
        'val_tasks_tr_features': val_tasks_tr_features,
        'val_tasks_tr_labels': val_tasks_tr_labels,
        'val_tasks_tr_corr': val_tasks_tr_corr,
        'val_tasks_test_features': val_tasks_test_features,
        'val_tasks_test_labels': val_tasks_test_labels,
        'val_tasks_test_corr': val_tasks_test_corr,
        # Test tasks
        'test_tasks_tr_features': test_tasks_tr_features,
        'test_tasks_tr_labels': test_tasks_tr_labels,
        'test_tasks_tr_corr': test_tasks_tr_corr,
        'test_tasks_test_features': test_tasks_test_features,
        'test_tasks_test_labels': test_tasks_test_labels,
        'test_tasks_test_corr': test_tasks_test_corr}
    return data


def concatenate_data(all_features, all_labels, all_corr):
    point_indexes_per_task = []
    for counter in range(len(all_features)):
        point_indexes_per_task.append(counter + np.zeros(all_features[counter].shape[0]))
    point_indexes_per_task = np.concatenate(point_indexes_per_task).astype(int)

    all_features = np.concatenate(all_features)
    all_labels = np.concatenate(all_labels)
    all_corr = np.concatenate(all_corr)
    return all_features, all_labels, all_corr, point_indexes_per_task


def split_tasks(all_features, indexes, all_labels=None, all_corr=None):
    # Split the blob/array of features into a list of tasks based on point_indexes_per_task
    all_features = [all_features[indexes == task_idx] for task_idx in np.unique(indexes)]
    if all_labels is None:
        return all_features
    all_labels = [all_labels[indexes == task_idx] for task_idx in np.unique(indexes)]
    all_corr = [all_corr[indexes == task_idx] for task_idx in np.unique(indexes)]
    return all_features, all_labels, all_corr

def select_tasks(tasks_indexes, test_features, test_labels, all_features, all_labels, pct=0.7, min_tasks=3):
    if not test_labels:
        return tasks_indexes
    score = np.zeros((len(test_features), len(tasks_indexes)))
    baseline = np.zeros(len(tasks_indexes))
    for i, ti in enumerate(tasks_indexes):
        mdl1 = LR()
        src_trials = int(all_features[ti].shape[0]*pct)
        x_tr_b = all_features[ti][:src_trials]
        y_tr_b = all_labels[ti][:src_trials]
        x_te = all_features[ti][src_trials:]
        y_te = all_labels[ti][src_trials:]
        mdl1.fit(x_tr_b, y_tr_b)
        baseline[i] = mdl1.score(x_te, y_te)
        for j in range(len(test_features)):
            mdl2 = LR()
            x_tr = np.concatenate((x_tr_b, test_features[j]))
            y_tr = np.concatenate((y_tr_b, test_labels[j]))
            mdl2.fit(x_tr, y_tr)
            score[j, i] = mdl2.score(x_te, y_te)

    mscore = np.mean(score, 0)
    diff = mscore > baseline
    if np.sum(diff) > min_tasks:
        return tasks_indexes[diff]
    else:
        return tasks_indexes # All the tasks reduce the performance, maybe we should not use transfer learning