import numpy as np
from numpy.linalg.linalg import norm
from src.data_management import DataHandler
from src.training import training


def main():
    seed = 999
    training_tasks_pct = 0.85
    validation_tasks_pct = 0.05
    test_tasks_pct = 0.1

    training_points_pct = 0.2
    validation_points_pct = 0.2
    test_points_pct = 0.6

    np.random.seed(seed)
    data_settings = {'seed': seed,
                     'training_tasks_pct': training_tasks_pct,
                     'validation_tasks_pct': validation_tasks_pct,
                     'test_tasks_pct': test_tasks_pct,
                     'training_points_pct': training_points_pct,
                     'validation_points_pct': validation_points_pct,
                     'test_points_pct': test_points_pct}

    ########################################################################
    ########################################################################
    ########################################################################
    # This chunk is hardcoded as an example of the structure the data should have.
    # A list of all the features and labels for all tasks basically
    n_tasks = 40
    dims = 10
    noise = 0.5
    all_features = []
    all_labels = []
    common_mean = 5 * np.random.randn(dims)
    # n_points = 60
    for task_idx in range(n_tasks):
        # Total number of points for the current task.
        n_points = np.random.randint(low=100, high=150)

        # Generating and normalizing the data.
        features = np.random.randn(n_points, dims)
        features = features / norm(features, axis=1, keepdims=True)
        # Generating the weight vector "around" the common mean.
        weight_vector = common_mean + np.random.randn(dims)
        # Linear model plus some noise.
        labels = features @ weight_vector + noise * np.random.randn(n_points)

        # Throwing the features and labels in their corresponding dictionary place.
        all_features.append(features)
        all_labels.append(labels)
    ########################################################################
    ########################################################################
    ########################################################################

    # print('ITL')
    # data = DataHandler(data_settings, all_features, all_labels)

    from sklearn.model_selection import train_test_split
    training_tasks_pct = data_settings['training_tasks_pct']
    validation_tasks_pct = data_settings['validation_tasks_pct']
    test_tasks_pct = data_settings['test_tasks_pct']
    training_tasks_indexes, temp_indexes = train_test_split(range(len(all_features)), test_size=1 - training_tasks_pct, shuffle=True)
    validation_tasks_indexes, test_tasks_indexes = train_test_split(temp_indexes, test_size=test_tasks_pct / (test_tasks_pct + validation_tasks_pct))

    training_tasks_training_features = [all_features[i] for i in training_tasks_indexes]
    training_tasks_training_labels = [all_labels[i] for i in training_tasks_indexes]
    point_indexes_per_training_task = [idx * np.ones(all_features[task_idx].shape[0]) for idx, task_idx in enumerate(training_tasks_indexes)]

    training_tasks_training_features = np.concatenate(training_tasks_training_features)
    training_tasks_training_labels = np.concatenate(training_tasks_training_labels)
    point_indexes_per_training_task = np.concatenate(point_indexes_per_training_task).astype(int)

    from src.ltl import BiasLTL

    # Optimizing metaparameters
    model_ltl = BiasLTL(regularization_parameter=1e-4, step_size_bit=1e+4)
    model_ltl.fit(training_tasks_training_features, training_tasks_training_labels, {'point_indexes_per_task': point_indexes_per_training_task})

    #############################################################################################################################
    validation_tasks_training_features = [all_features[i] for i in validation_tasks_indexes]
    validation_tasks_training_labels = [all_labels[i] for i in validation_tasks_indexes]
    point_indexes_per_validation_task = [idx * np.ones(all_features[task_idx].shape[0]) for idx, task_idx in enumerate(validation_tasks_indexes)]

    # Checking performance on validation tasks
    validation_tasks_training_features = np.concatenate(validation_tasks_training_features)
    validation_tasks_training_labels = np.concatenate(validation_tasks_training_labels)
    point_indexes_per_validation_task = np.concatenate(point_indexes_per_validation_task).astype(int)
    weight_vectors_per_task = model_ltl.fit_inner(validation_tasks_training_features,
                                                  all_labels=validation_tasks_training_labels,
                                                  extra_inputs={'predictions_for_each_training_task': False,
                                                                'point_indexes_per_task': point_indexes_per_validation_task})
    #############################################################################################################################
    validation_tasks_test_features = [all_features[i] for i in validation_tasks_indexes]
    validation_tasks_test_labels = [all_labels[i] for i in validation_tasks_indexes]
    point_indexes_per_validation_task = [idx * np.ones(all_features[task_idx].shape[0]) for idx, task_idx in enumerate(validation_tasks_indexes)]

    validation_tasks_test_features = np.concatenate(validation_tasks_test_features)
    point_indexes_per_validation_task = np.concatenate(point_indexes_per_validation_task).astype(int)
    predictions = model_ltl.predict(validation_tasks_test_features, weight_vectors_per_task,
                                    extra_inputs={'predictions_for_each_training_task': False,
                                                  'point_indexes_per_task': point_indexes_per_validation_task})

    # from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer

    def metalearning_mse(all_labels, all_predictions, point_indexes_per_task=None):
        if point_indexes_per_task is None:
            raise ValueError("Required input task_indexes not passed.")

        # Take lists and return errors
        error = None
        return error

    metalearning_mse = make_scorer(metalearning_mse, greater_is_better=False, needs_proba=False, needs_threshold=False, point_indexes_per_task=point_indexes_per_training_task)
    # Extra temporary column used as task identifier
    features = np.append(features, task_idx * np.ones((features.shape[0], 1)), axis=1)
    #
    all_features = np.concatenate(all_features)
    all_labels = np.concatenate(all_labels)

    training_settings_itl = {'regularization_parameter_range': [10 ** float(i) for i in np.linspace(-12, 4, 20)],
                             'method': 'ITL'}

    results_itl = training(data, training_settings_itl)
    ###########
    # print('\nBatch LTL')
    # training_settings_batch_ltl = {'regularization_parameter_range': [10 ** float(i) for i in np.linspace(-12, 4, 30)],
    #                                'method': 'batch_LTL'}
    #
    # results_batch_ltl = training(data, training_settings_batch_ltl)
    ###########
    print('\nOnline LTL')
    training_settings_online_ltl = {'regularization_parameter_range': [10 ** float(i) for i in np.linspace(-12, 4, 20)],
                                    'step_size': 1e+3,
                                    'method': 'online_LTL'}

    results_online_ltl = training(data, training_settings_online_ltl)

    import matplotlib.pyplot as plt
    plt.axhline(y=results_itl['test_perfomance'], xmin=0, xmax=len(all_features) - 1, color='k', label='indipendent learning')
    # plt.axhline(y=results_batch_ltl['best_test_performance'], xmin=0, xmax=len(all_features)-1, color='tab:red', label='batch ltl')
    plt.plot(results_online_ltl['best_test_performances'], color='tab:blue', label='online ltl')
    plt.legend()
    plt.xlabel('number of training tasks')
    plt.ylabel('performanance')
    plt.pause(0.1)
    plt.show()
    print('done')


if __name__ == "__main__":
    main()
