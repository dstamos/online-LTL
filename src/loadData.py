import numpy as np

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
    return feat, label


def split_data_essex(all_features, all_labels, train, retrain, test):
    """
    Training tasks only have training data.
    Validation tasks only have training and test data.
    Test tasks have training, validation and test data. The validation data are currently not used for metalearning, only for independent learning.
    :param all_features: list of numpy arrays (n_points, dims)
    :param all_labels:  list of numpy arrays (n_points, )
    :param data_settings:  dict of settings
    :return:
    """
    training_tasks_indexes = train
    validation_tasks_indexes = retrain
    test_tasks_indexes = test

    # Training tasks (only training data)
    training_tasks_training_features = [all_features[i] for i in training_tasks_indexes]
    training_tasks_training_labels = [all_labels[i] for i in training_tasks_indexes]

    # Validation tasks (training and test data)
    re_train_features = [all_features[i] for i in validation_tasks_indexes]
    re_train_labels = [all_labels[i] for i in validation_tasks_indexes]

    # Test tasks (training, validation and test tasks)
    test_tasks_training_features = [all_features[i] for i in test_tasks_indexes]
    test_tasks_training_labels = [all_labels[i] for i in test_tasks_indexes]

    data = {'training_tasks_indexes': training_tasks_indexes,
            'validation_tasks_indexes': validation_tasks_indexes,
            'test_tasks_indexes': test_tasks_indexes,
            # Training tasks
            'training_tasks_training_features': training_tasks_training_features,
            'training_tasks_training_labels': training_tasks_training_labels,
            # Validation tasks
            'validation_tasks_training_features': re_train_features,
            'validation_tasks_training_labels': re_train_labels,
            # Test tasks
            'test_tasks_training_features': test_tasks_training_features,
            'test_tasks_training_labels': test_tasks_training_labels
            }
    return data
