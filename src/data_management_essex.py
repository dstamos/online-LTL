import numpy as np
from sklearn.model_selection import train_test_split


def load_data_essex_one(delete0=True, useStim=True, useRT=True):
    extra = np.load('./data/extra.npy')
    correct = np.load('./data/corr.npy')
    if useStim:
        stim = np.load('./data/stimFeatures.npy')
    resp = np.load('./data/respFeatures.npy')
    feat = []
    label = []
    corr = []
    for s in np.unique(extra[:, 0]):
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


def load_data_essex_two():

    l = np.load('./data/alllabels_chris.npy', allow_pickle=True)
    f = np.load('./data/allfeatures_chris.npy', allow_pickle=True)
    c = np.load('./data/correctness.npy', allow_pickle=True)
    features = [i for i in f]
    labels = [i for i in l]
    corr = [i.astype(bool) for i in c]

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

    test_tasks_indexes = []
    for test_subject in test_subjects:
        curr_indexes = test_subject * n_experiments_per_subject + np.arange(0, 3)
        test_tasks_indexes = test_tasks_indexes + curr_indexes.tolist()

    tasks_indexes = list(range(0, n_all_subjects * n_experiments_per_subject))
    for idx in test_tasks_indexes:
        tasks_indexes.remove(idx)
    # Validation tasks are picked randomly (not from the same person)
    if settings['merge_test']:
        list_of_subjects = list(range(n_all_subjects))
        list_without_test_subjects = [s for s in list_of_subjects if s not in test_subjects]
        validation_tasks_indexes = np.random.choice(list_without_test_subjects) * n_experiments_per_subject + np.arange(0, 3)
        for idx in validation_tasks_indexes:
            tasks_indexes.remove(idx)
        training_tasks_indexes = tasks_indexes
    else:
        training_tasks_indexes, validation_tasks_indexes = train_test_split(tasks_indexes, test_size=n_experiments_per_subject)

    tr_tasks_tr_points_pct = settings['tr_tasks_tr_points_pct']
    val_tasks_tr_points_pct = settings['val_tasks_tr_points_pct']
    val_tasks_test_points_pct = settings['val_tasks_test_points_pct']
    test_tasks_tr_points_pct = settings['test_tasks_tr_points_pct']
    test_tasks_test_points_pct = settings['test_tasks_test_points_pct']

    # Training tasks (only training data)
    tr_tasks_tr_features = []
    tr_tasks_tr_labels = []
    tr_tasks_tr_corr = []
    for counter, task_index in enumerate(training_tasks_indexes):
        x = all_features[task_index]
        y = all_labels[task_index]
        corr = all_corr[task_index]
        n_all_points = len(y)
        shuffled_points_indexes = np.random.permutation(range(n_all_points))
        n_tr_points = int(tr_tasks_tr_points_pct * n_all_points)
        training_features = x[shuffled_points_indexes[:n_tr_points], :]
        training_labels = y[shuffled_points_indexes[:n_tr_points]]
        training_corr = corr[shuffled_points_indexes[:n_tr_points]]

        tr_tasks_tr_features.append(training_features)
        tr_tasks_tr_labels.append(training_labels)
        tr_tasks_tr_corr.append(training_corr)

        if verbose is True:
            print(f'task: {all_experiment_names[task_index]:s} ({task_index:2d}) | points: {n_all_points:4d} | tr: {n_tr_points:4d}')

    # Validation tasks (training and test data)
    val_tasks_tr_features = []
    val_tasks_tr_labels = []
    val_tasks_tr_corr = []
    val_tasks_test_features = []
    val_tasks_test_labels = []
    val_tasks_test_corr = []
    if settings['merge_test']:
        x = np.zeros((0, all_features[0].shape[1]))
        y = np.zeros(0)
        corr = np.zeros(0, bool)
        for task_index in validation_tasks_indexes:
            x = np.concatenate((x, all_features[task_index]), 0)
            y = np.concatenate((y, all_labels[task_index]), 0)
            corr = np.concatenate((corr, all_corr[task_index]), 0)
        validation_tasks_indexes = [0]  # There is only one training task
        n_all_points = len(y)
        n_tr_points = int(val_tasks_tr_points_pct * n_all_points)

        training_features = x[:n_tr_points, :]
        training_labels = y[:n_tr_points]
        training_corr = corr[:n_tr_points]
        test_features = x[n_tr_points:, :]
        test_labels = y[n_tr_points:]
        test_corr = corr[n_tr_points:]

        val_tasks_tr_features.append(training_features)
        val_tasks_tr_labels.append(training_labels)
        val_tasks_tr_corr.append(training_corr)
        val_tasks_test_features.append(test_features)
        val_tasks_test_labels.append(test_labels)
        val_tasks_test_corr.append(test_corr)
    else:
        for counter, task_index in enumerate(validation_tasks_indexes):
            x = all_features[task_index]
            y = all_labels[task_index]
            corr = all_corr[task_index]
            n_all_points = len(y)
            flag = True
            while flag:
                shuffled_points_indexes = np.random.permutation(range(n_all_points))
                n_tr_points = int(val_tasks_tr_points_pct * n_all_points)
                test_corr = corr[shuffled_points_indexes[n_tr_points:]]
                training_corr = corr[shuffled_points_indexes[:n_tr_points]]
                if (np.sum(test_corr) != len(test_corr) and np.sum(training_corr) != len(training_corr)) or n_tr_points==0:
                    flag = False

            training_features = x[shuffled_points_indexes[:n_tr_points], :]
            training_labels = y[shuffled_points_indexes[:n_tr_points]]
            test_features = x[shuffled_points_indexes[n_tr_points:], :]
            test_labels = y[shuffled_points_indexes[n_tr_points:]]

            val_tasks_tr_features.append(training_features)
            val_tasks_tr_labels.append(training_labels)
            val_tasks_tr_corr.append(training_corr)
            val_tasks_test_features.append(test_features)
            val_tasks_test_labels.append(test_labels)
            val_tasks_test_corr.append(test_corr)

            if verbose is True:
                print(f'task: {all_experiment_names[task_index]:s} ({task_index:2d}) | points: {n_all_points:4d} | tr: {n_tr_points:4d} | test: {n_all_points - n_tr_points:4d}')

    # Test tasks (training and test data)
    test_tasks_tr_features = []
    test_tasks_tr_labels = []
    test_tasks_tr_corr = []
    test_tasks_test_features = []
    test_tasks_test_labels = []
    test_tasks_test_corr = []
    if settings['merge_test']:
        x = np.zeros((0, all_features[0].shape[1]))
        y = np.zeros(0)
        corr = np.zeros(0, bool)
        for task_index in test_tasks_indexes:
            x = np.concatenate((x, all_features[task_index]), 0)
            y = np.concatenate((y, all_labels[task_index]), 0)
            corr = np.concatenate((corr, all_corr[task_index]), 0)
        test_tasks_indexes = [0]# There is only one training task
        n_all_points = len(y)
        n_tr_points = int(test_tasks_tr_points_pct * n_all_points)

        training_features = x[:n_tr_points, :]
        training_labels = y[:n_tr_points]
        training_corr = corr[:n_tr_points]
        test_features = x[n_tr_points:, :]
        test_labels = y[n_tr_points:]
        test_corr = corr[n_tr_points:]

        test_tasks_tr_features.append(training_features)
        test_tasks_tr_labels.append(training_labels)
        test_tasks_tr_corr.append(training_corr)
        test_tasks_test_features.append(test_features)
        test_tasks_test_labels.append(test_labels)
        if all_corr:
            test_tasks_test_corr.append(test_corr)

    else:
        for task_index in test_tasks_indexes:
            x = all_features[task_index]
            y = all_labels[task_index]
            corr = all_corr[task_index]
            n_all_points = len(y)
            n_tr_points = int(test_tasks_tr_points_pct * n_all_points)
            flag = True
            while flag:
                shuffled_points_indexes = np.random.permutation(range(n_all_points))
                n_tr_points = int(test_tasks_tr_points_pct * n_all_points)
                test_corr = corr[shuffled_points_indexes[n_tr_points:]]
                training_corr = corr[shuffled_points_indexes[:n_tr_points]]
                if (np.sum(test_corr) != len(test_corr) and np.sum(training_corr) != len(training_corr)) or n_tr_points == 0:
                    flag = False

            training_features = x[shuffled_points_indexes[:n_tr_points], :]
            training_labels = y[shuffled_points_indexes[:n_tr_points]]
            test_features = x[shuffled_points_indexes[n_tr_points:], :]
            test_labels = y[shuffled_points_indexes[n_tr_points:]]

            test_tasks_tr_features.append(training_features)
            test_tasks_tr_labels.append(training_labels)
            test_tasks_tr_corr.append(training_corr)
            test_tasks_test_features.append(test_features)
            test_tasks_test_labels.append(test_labels)
            test_tasks_test_corr.append(test_corr)

            if verbose is True:
                print(f'task: {all_experiment_names[task_index]:s} ({task_index:2d}) | points: {n_all_points:4d} | tr: {n_tr_points:4d} | test: {n_all_points - n_tr_points:4d}')
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
