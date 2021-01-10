import numpy as np
from sklearn.model_selection import train_test_split


def load_data_essex(delete0=True, useStim=True, useRT=True):
    extra = np.load('./data/extra.npy')
    if useStim:
        stim = np.load('./data/stimFeatures.npy')
    resp = np.load('./data/respFeatures.npy')
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
    n_all_subjects = len(all_features) // n_experiments_per_subject  # Hardcoded - the assumption is that all subjects had 3 days of experiments
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

    tr_tasks_tr_points_pct = settings['tr_tasks_tr_points_pct']
    val_tasks_tr_points_pct = settings['val_tasks_tr_points_pct']
    val_tasks_val_points_pct = settings['val_tasks_val_points_pct']
    test_tasks_tr_points_pct = settings['test_tasks_tr_points_pct']
    test_tasks_val_points_pct = settings['test_tasks_val_points_pct']
    test_tasks_test_points_pct = settings['test_tasks_test_points_pct']

    # Training tasks (only training data)
    tr_tasks_tr_features = []
    tr_tasks_tr_labels = []
    for counter, task_index in enumerate(training_tasks_indexes):
        x = all_features[task_index]
        y = all_labels[task_index]
        n_all_points = len(y)
        shuffled_points_indexes = np.random.permutation(range(n_all_points))
        n_tr_points = int(tr_tasks_tr_points_pct * n_all_points)
        training_features = x[shuffled_points_indexes[:n_tr_points], :]
        training_labels = y[shuffled_points_indexes[:n_tr_points]]

        tr_tasks_tr_features.append(training_features)
        tr_tasks_tr_labels.append(training_labels)

        print(f'task: {all_experiment_names[task_index]:s} ({task_index:2d}) | points: {n_all_points:4d} | tr: {n_tr_points:4d}')

    # Validation tasks (training and test data)
    val_tasks_tr_features = []
    val_tasks_tr_labels = []
    val_tasks_val_features = []
    val_tasks_val_labels = []
    for counter, task_index in enumerate(validation_tasks_indexes):
        x = all_features[task_index]
        y = all_labels[task_index]
        n_all_points = len(y)
        shuffled_points_indexes = np.random.permutation(range(n_all_points))
        n_tr_points = int(val_tasks_tr_points_pct * n_all_points)
        n_val_points = int(val_tasks_val_points_pct * n_all_points)
        training_features = x[shuffled_points_indexes[:n_tr_points], :]
        training_labels = y[shuffled_points_indexes[:n_tr_points]]
        validation_features = x[shuffled_points_indexes[n_tr_points + 1:n_tr_points + n_val_points], :]
        validation_labels = y[shuffled_points_indexes[n_tr_points + 1:n_tr_points + n_val_points]]

        val_tasks_tr_features.append(training_features)
        val_tasks_tr_labels.append(training_labels)
        val_tasks_val_features.append(validation_features)
        val_tasks_val_labels.append(validation_labels)

        print(f'task: {all_experiment_names[task_index]:s} ({task_index:2d}) | points: {n_all_points:4d} | tr: {n_tr_points:4d} | val: {n_val_points:4d}')

    # Test tasks (training, validation and test data)
    test_tasks_tr_features = []
    test_tasks_tr_labels = []
    test_tasks_val_features = []
    test_tasks_val_labels = []
    test_tasks_test_features = []
    test_tasks_test_labels = []
    for counter, task_index in enumerate(test_tasks_indexes):
        x = all_features[task_index]
        y = all_labels[task_index]
        n_all_points = len(y)
        shuffled_points_indexes = np.random.permutation(range(n_all_points))
        n_tr_points = int(test_tasks_tr_points_pct * n_all_points)
        n_val_points = int(test_tasks_val_points_pct * n_all_points)
        n_test_points = int(test_tasks_test_points_pct * n_all_points)
        training_features = x[shuffled_points_indexes[:n_tr_points], :]
        training_labels = y[shuffled_points_indexes[:n_tr_points]]
        validation_features = x[shuffled_points_indexes[n_tr_points + 1:n_tr_points + n_val_points], :]
        validation_labels = y[shuffled_points_indexes[n_tr_points + 1:n_tr_points + n_val_points]]
        test_features = x[shuffled_points_indexes[n_tr_points + n_val_points + 1:n_tr_points + n_val_points + n_test_points], :]
        test_labels = y[shuffled_points_indexes[n_tr_points + n_val_points + 1:n_tr_points + n_val_points + n_test_points]]

        test_tasks_tr_features.append(training_features)
        test_tasks_tr_labels.append(training_labels)
        test_tasks_val_features.append(validation_features)
        test_tasks_val_labels.append(validation_labels)
        test_tasks_test_features.append(test_features)
        test_tasks_test_labels.append(test_labels)

        print(f'task: {all_experiment_names[task_index]:s} ({task_index:2d}) | points: {n_all_points:4d} | tr: {n_tr_points:4d} | val: {n_val_points:4d} | test: {n_test_points:4d}')

    data = {'training_tasks_indexes': training_tasks_indexes,
            'validation_tasks_indexes': validation_tasks_indexes,
            'test_tasks_indexes': test_tasks_indexes,
            # Training tasks
            'tr_tasks_tr_features': tr_tasks_tr_features,
            'tr_tasks_tr_labels': tr_tasks_tr_labels,
            # Validation tasks
            'val_tasks_tr_features': val_tasks_tr_features,
            'val_tasks_tr_labels': val_tasks_tr_labels,
            'val_tasks_val_features': val_tasks_val_features,
            'val_tasks_val_labels': val_tasks_val_labels,
            # Test tasks
            'test_tasks_tr_features': test_tasks_tr_features,
            'test_tasks_tr_labels': test_tasks_tr_labels,
            'test_tasks_val_features': test_tasks_val_features,
            'test_tasks_val_labels': test_tasks_val_labels,
            'test_tasks_test_features': test_tasks_test_features,
            'test_tasks_test_labels': test_tasks_test_labels}
    return data


def concatenate_data(all_features, all_labels):
    point_indexes_per_task = []
    for counter in range(len(all_features)):
        point_indexes_per_task.append(counter + np.zeros(all_features[counter].shape[0]))
    point_indexes_per_task = np.concatenate(point_indexes_per_task).astype(int)

    all_features = np.concatenate(all_features)
    all_labels = np.concatenate(all_labels)
    return all_features, all_labels, point_indexes_per_task


def split_tasks(all_features, indexes, all_labels=None):
    # Split the blob/array of features into a list of tasks based on point_indexes_per_task
    all_features = [all_features[indexes == task_idx] for task_idx in np.unique(indexes)]
    if all_labels is None:
        return all_features
    all_labels = [all_labels[indexes == task_idx] for task_idx in np.unique(indexes)]
    return all_features, all_labels
