from sklearn.model_selection import train_test_split
import numpy as np


def split_data(all_features, all_labels, data_settings):
    """
    Training tasks only have training data.
    Validation tasks only have training and test data.
    Test tasks have training, validation and test data. The validation data are currently not used for metalearning, only for independent learning.
    :param all_features: list of numpy arrays (n_points, dims)
    :param all_labels:  list of numpy arrays (n_points, )
    :param data_settings:  dict of settings
    :return:
    """
    training_tasks_pct = data_settings['training_tasks_pct']
    validation_tasks_pct = data_settings['validation_tasks_pct']
    test_tasks_pct = data_settings['test_tasks_pct']
    training_tasks_indexes, temp_indexes = train_test_split(range(len(all_features)), test_size=1 - training_tasks_pct, shuffle=True)
    validation_tasks_indexes, test_tasks_indexes = train_test_split(temp_indexes, test_size=test_tasks_pct / (test_tasks_pct + validation_tasks_pct))

    training_points_pct = data_settings['training_points_pct']
    validation_points_pct = data_settings['validation_points_pct']
    test_points_pct = data_settings['test_points_pct']

    # Training tasks (only training data)
    training_tasks_training_features = [all_features[i] for i in training_tasks_indexes]
    training_tasks_training_labels = [all_labels[i] for i in training_tasks_indexes]

    # Validation tasks (training and test data)
    validation_tasks_training_features = []
    validation_tasks_training_labels = []
    validation_tasks_test_features = []
    validation_tasks_test_labels = []
    for counter, task_index in enumerate(validation_tasks_indexes):
        training_features, test_features, training_labels, test_labels = train_test_split(all_features[task_index], all_labels[task_index], test_size=1 - training_points_pct, shuffle=True)

        validation_tasks_training_features.append(training_features)
        validation_tasks_training_labels.append(training_labels)
        validation_tasks_test_features.append(test_features)
        validation_tasks_test_labels.append(test_labels)

    # Test tasks (training, validation and test tasks)
    test_tasks_training_features = []
    test_tasks_training_labels = []
    test_tasks_validation_features = []
    test_tasks_validation_labels = []
    test_tasks_test_features = []
    test_tasks_test_labels = []
    for counter, task_index in enumerate(test_tasks_indexes):
        training_features, temp_features, training_labels, temp_labels = train_test_split(all_features[task_index], all_labels[task_index], test_size=1 - training_points_pct, shuffle=True)
        validation_features, test_features, validation_labels, test_labels = train_test_split(temp_features, temp_labels, test_size=test_points_pct / (test_points_pct + validation_points_pct))

        test_tasks_training_features.append(training_features)
        test_tasks_training_labels.append(training_labels)
        test_tasks_validation_features.append(validation_features)
        test_tasks_validation_labels.append(validation_labels)
        test_tasks_test_features.append(test_features)
        test_tasks_test_labels.append(test_labels)

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


def concatenate_data(all_features, all_labels):
    point_indexes_per_task = []
    for counter in range(len(all_features)):
        point_indexes_per_task.append(counter + np.zeros(all_features[counter].shape[0]))
    point_indexes_per_task = np.concatenate(point_indexes_per_task).astype(int)

    all_features = np.concatenate(all_features)
    all_labels = np.concatenate(all_labels)
    return all_features, all_labels, point_indexes_per_task
