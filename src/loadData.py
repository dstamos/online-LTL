import numpy as np

def load_data_essex(path='', delete0=True, useStim=True, useRT=True, addBias=True):
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
            if addBias:
                f = np.concatenate((np.ones((nval, 1)), f), 1)
            if useStim:
                f = np.concatenate((f, stim[val, :]), 1)
            if useRT:
                f = np.concatenate((f, np.expand_dims(extra[val, 2], 1)), 1)
            feat.append(f)
            label.append(extra[val, 3])
    return feat, label

def split_data_essex(all_features, all_labels, retrain, test):
    """
    Training tasks only have training data.
    Validation tasks only have training and test data.
    Test tasks have training, validation and test data. The validation data are currently not used for metalearning, only for independent learning.
    :param all_features: list of numpy arrays (n_points, dims)
    :param all_labels:  list of numpy arrays (n_points, )
    :param data_settings:  dict of settings
    :return:
    """
    indexes = np.arange(len(all_features))

    training_tasks_indexes = np.setdiff1d(indexes, retrain)
    training_tasks_indexes = np.setdiff1d(training_tasks_indexes, test)
    test_tasks_indexes = test
    validation_tasks_indexes = retrain

    # Training tasks (only training data)
    training_tasks_training_features = [all_features[i] for i in  training_tasks_indexes]
    training_tasks_training_labels = [all_labels[i] for i in training_tasks_indexes]

    # Validation tasks (training and test data)
    validation_tasks_training_features = [all_features[i] for i in validation_tasks_indexes]
    validation_tasks_training_labels = [all_labels[i] for i in validation_tasks_indexes]

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
            'validation_tasks_training_features': validation_tasks_training_features,
            'validation_tasks_training_labels': validation_tasks_training_labels,
            # Test tasks
            'test_tasks_training_features': test_tasks_training_features,
            'test_tasks_training_labels': test_tasks_training_labels
            }
    return data