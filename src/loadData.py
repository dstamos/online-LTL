import numpy as np
from sklearn.model_selection import train_test_split


def load_data_essex(path='', delete0=True, useStim=True, useRT=True):
    extra = np.load(path+'extra.npy')
    if useStim:
        stim = np.load(path + 'stimFeatures.npy')
    resp = np.load(path + 'respFeatures.npy')
    feat = []
    label = []
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

    return feat, label, experiment_names


def split_data_essex(all_features, all_labels, all_experiment_names, settings):
    """
    Training tasks only have training data.
    Validation tasks only have training and test data.
    Test tasks have training, validation and test data. The validation data are currently not used for metalearning, only for independent learning.
    :param all_experiment_names: list of experiment names
    :param all_features: list of numpy arrays (n_points, dims)
    :param all_labels:  list of numpy arrays (n_points, )
    :param settings:  dict of settings
    :return:
    """

    """
    This function is very much hardcoded for the setting in which we have T subjects with 3 experiments per subject (3 * T total experiments/tasks). 
    We leave n_test_subjects subjects out for testing (3 experiments for each), and 3 experiments out for validation (they can belong to ANY subject).
    """
    # FIXME A bunch of hardcoded things in this function.
    # Pick the test_subjects, find the corresponding test_tasks_indexes.
    n_experiments_per_subject = 3
    n_all_subjects = len(all_features) // n_experiments_per_subject   # Hardcoded - the assumption is that all subjects had 3 days of experiments
    n_test_subjects = settings['n_test_subjects']
    test_subjects = np.random.choice(range(n_all_subjects), size=n_test_subjects, replace=False)

    test_tasks_indexes = []
    for test_subject in test_subjects:
        curr_indexes = test_subject * n_experiments_per_subject + np.arange(0, 3)
        test_tasks_indexes = test_tasks_indexes + curr_indexes.tolist()

    tasks_indexes = list(range(0, n_all_subjects * n_experiments_per_subject))
    for idx in test_tasks_indexes:
        tasks_indexes.remove(idx)
    training_tasks_indexes, validation_tasks_indexes = train_test_split(tasks_indexes, test_size=n_experiments_per_subject)

    training_tasks_training_points_pct = settings['training_tasks_training_points_pct']
    validation_tasks_training_points_pct = settings['validation_tasks_training_points_pct']
    validation_tasks_validation_points_pct = settings['validation_tasks_validation_points_pct']
    test_tasks_training_points_pct = settings['test_tasks_training_points_pct']
    test_tasks_validation_points_pct = settings['test_tasks_validation_points_pct']
    test_tasks_test_points_pct = settings['test_tasks_test_points_pct']

    # Training tasks (only training data)
    training_tasks_training_features = []
    training_tasks_training_labels = []
    for counter, task_index in enumerate(training_tasks_indexes):
        x = all_features[task_index]
        y = all_labels[task_index]
        n_all_points = len(y)
        shuffled_points_indexes = np.random.permutation(range(n_all_points))
        n_tr_points = int(training_tasks_training_points_pct * n_all_points)
        training_features = x[shuffled_points_indexes[:n_tr_points], :]
        training_labels = y[shuffled_points_indexes[:n_tr_points]]

        training_tasks_training_features.append(training_features)
        training_tasks_training_labels.append(training_labels)

        print(f'task: {all_experiment_names[task_index]:s} ({task_index:2d}) | points: {n_all_points:4d} | tr points: {n_tr_points:5}')

    # Validation tasks (training and test data)
    validation_tasks_training_features = []
    validation_tasks_training_labels = []
    validation_tasks_test_features = []
    validation_tasks_test_labels = []
    for counter, task_index in enumerate(validation_tasks_indexes):
        x = all_features[task_index]
        y = all_labels[task_index]
        n_all_points = len(y)
        shuffled_points_indexes = np.random.permutation(range(n_all_points))
        n_tr_points = int(validation_tasks_training_points_pct * n_all_points)
        n_val_points = int(validation_tasks_validation_points_pct * n_all_points)
        training_features = x[shuffled_points_indexes[:n_tr_points], :]
        training_labels = y[shuffled_points_indexes[:n_tr_points]]
        validation_features = x[shuffled_points_indexes[n_tr_points+1:n_tr_points+n_val_points], :]
        validation_labels = y[shuffled_points_indexes[n_tr_points+1:n_tr_points+n_val_points]]

        validation_tasks_training_features.append(training_features)
        validation_tasks_training_labels.append(training_labels)
        validation_tasks_test_features.append(validation_features)
        validation_tasks_test_labels.append(validation_labels)

        print(f'task: {all_experiment_names[task_index]:s} ({task_index:2d}) | points: {n_all_points:4d} | tr points: {n_tr_points:4d} | val points: {n_val_points:4d}')

    # Test tasks (training, validation and test data)
    test_tasks_training_features = []
    test_tasks_training_labels = []
    test_tasks_validation_features = []
    test_tasks_validation_labels = []
    test_tasks_test_features = []
    test_tasks_test_labels = []
    for counter, task_index in enumerate(test_tasks_indexes):
        x = all_features[task_index]
        y = all_labels[task_index]
        n_all_points = len(y)
        shuffled_points_indexes = np.random.permutation(range(n_all_points))
        n_tr_points = int(validation_tasks_training_points_pct * n_all_points)
        n_val_points = int(validation_tasks_validation_points_pct * n_all_points)
        n_test_points = int(test_tasks_test_points_pct * n_all_points)
        training_features = x[shuffled_points_indexes[:n_tr_points], :]
        training_labels = y[shuffled_points_indexes[:n_tr_points]]
        validation_features = x[shuffled_points_indexes[n_tr_points+1:n_tr_points+n_val_points], :]
        validation_labels = y[shuffled_points_indexes[n_tr_points+1:n_tr_points+n_val_points]]
        test_features = x[shuffled_points_indexes[n_tr_points+n_val_points+1:n_tr_points+n_val_points+n_test_points], :]
        test_labels = y[shuffled_points_indexes[n_tr_points+n_val_points+1:n_tr_points+n_val_points+n_test_points]]

        test_tasks_training_features.append(training_features)
        test_tasks_training_labels.append(training_labels)
        test_tasks_validation_features.append(validation_features)
        test_tasks_validation_labels.append(validation_labels)
        test_tasks_test_features.append(test_features)
        test_tasks_test_labels.append(test_labels)

        print(f'task: {all_experiment_names[task_index]:s} ({task_index:2d}) | points: {n_all_points:4d} | tr points: {n_tr_points:4d} | val points: {n_val_points:4d} | ts points: {n_test_points:4d}')

    data = {'training_tasks_indexes': training_tasks_indexes,
            'validation_tasks_indexes': validation_tasks_indexes,
            'test_tasks_indexes': test_tasks_indexes,
            # Training tasks
            'training_tasks_training_features': training_tasks_training_features,
            'training_tasks_training_labels': training_tasks_training_labels,
            # Validation tasks
            'validation_tasks_training_features': validation_tasks_training_features,
            'validation_tasks_training_labels': validation_tasks_training_labels,
            'validation_tasks_test_features': validation_tasks_test_features,
            'validation_tasks_test_labels': validation_tasks_test_labels,
            # Test tasks
            'test_tasks_training_features': test_tasks_training_features,
            'test_tasks_training_labels': test_tasks_training_labels,
            'test_tasks_validation_features': test_tasks_validation_features,
            'test_tasks_validation_labels': test_tasks_validation_labels,
            'test_tasks_test_features': test_tasks_test_features,
            'test_tasks_test_labels': test_tasks_test_labels}
    return data
