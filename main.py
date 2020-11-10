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
    n_tasks = 300
    dims = 10
    noise = 0.5
    all_features = []
    all_labels = []
    common_mean = 5 * np.random.randn(dims)
    n_points = 60
    for task_idx in range(n_tasks):
        # Total number of points for the current task.
        # n_points = np.random.randint(low=100, high=150)

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

    from src.ltl import BiasLTL
    from src.utilities import multiple_tasks_mse
    from sklearn.model_selection import train_test_split

    training_tasks_pct = data_settings['training_tasks_pct']
    validation_tasks_pct = data_settings['validation_tasks_pct']
    test_tasks_pct = data_settings['test_tasks_pct']
    training_tasks_indexes, temp_indexes = train_test_split(range(len(all_features)), test_size=1 - training_tasks_pct, shuffle=True)
    validation_tasks_indexes, test_tasks_indexes = train_test_split(temp_indexes, test_size=test_tasks_pct / (test_tasks_pct + validation_tasks_pct))

    # Training task data
    training_tasks_training_features = [all_features[i] for i in training_tasks_indexes]
    training_tasks_training_labels = [all_labels[i] for i in training_tasks_indexes]
    point_indexes_per_training_task = [idx * np.ones(all_features[task_idx].shape[0]) for idx, task_idx in enumerate(training_tasks_indexes)]

    training_tasks_training_features = np.concatenate(training_tasks_training_features)
    training_tasks_training_labels = np.concatenate(training_tasks_training_labels)
    point_indexes_per_training_task = np.concatenate(point_indexes_per_training_task).astype(int)

    # Validation task data
    validation_tasks_training_features = [all_features[i] for i in validation_tasks_indexes]
    validation_tasks_training_labels = [all_labels[i] for i in validation_tasks_indexes]
    point_indexes_per_validation_task = [idx * np.ones(all_features[task_idx].shape[0]) for idx, task_idx in enumerate(validation_tasks_indexes)]

    validation_tasks_training_features = np.concatenate(validation_tasks_training_features)
    validation_tasks_training_labels = np.concatenate(validation_tasks_training_labels)
    point_indexes_per_validation_task = np.concatenate(point_indexes_per_validation_task).astype(int)

    validation_tasks_test_features = [all_features[i] for i in validation_tasks_indexes]
    validation_tasks_test_labels = [all_labels[i] for i in validation_tasks_indexes]
    validation_tasks_test_features = np.concatenate(validation_tasks_test_features)
    # Test task data
    test_tasks_training_features = [all_features[i] for i in test_tasks_indexes]
    test_tasks_training_labels = [all_labels[i] for i in test_tasks_indexes]
    point_indexes_per_test_task = [idx * np.ones(all_features[task_idx].shape[0]) for idx, task_idx in enumerate(test_tasks_indexes)]

    test_tasks_training_features = np.concatenate(test_tasks_training_features)
    test_tasks_training_labels = np.concatenate(test_tasks_training_labels)
    point_indexes_per_test_task = np.concatenate(point_indexes_per_test_task).astype(int)

    test_tasks_test_features = [all_features[i] for i in test_tasks_indexes]
    test_tasks_test_labels = [all_labels[i] for i in test_tasks_indexes]
    test_tasks_test_features = np.concatenate(test_tasks_test_features)
    """
    Optimize metaparameter on the training data of the training tasks
    """
    model_ltl = BiasLTL(regularization_parameter=1e-1, step_size_bit=1e+3)
    model_ltl.fit(training_tasks_training_features, training_tasks_training_labels, {'point_indexes_per_task': point_indexes_per_training_task})

    """
    Optimize the weight vectors on the training data of the target tasks (those should be training data from the validation or test tasks)
    Passing predictions_for_each_training_task as True, recovers the weight vectors for each metaparameter that was returned during the training (one for each training task).        
    """
    extra_inputs = {'predictions_for_each_training_task': False, 'point_indexes_per_task': point_indexes_per_validation_task}
    weight_vectors_per_task = model_ltl.fit_inner(validation_tasks_training_features,
                                                  all_labels=validation_tasks_training_labels,
                                                  extra_inputs=extra_inputs)
    """
    Take the weight vectors from the previous step and make predictions on the test data of the same tasks you optimized them on
    """

    predictions_validation = model_ltl.predict(validation_tasks_test_features, weight_vectors_per_task, extra_inputs=extra_inputs)

    val_performance = multiple_tasks_mse(validation_tasks_test_labels, predictions_validation, extra_inputs['predictions_for_each_training_task'])
    print(val_performance)

    """
    Optimize the weight vectors on the training data of the target tasks (those should be training data from the validation or test tasks)
    Passing predictions_for_each_training_task as True, recovers the weight vectors for each metaparameter that was returned during the training (one for each training task).        
    """
    extra_inputs = {'predictions_for_each_training_task': True, 'point_indexes_per_task': point_indexes_per_test_task}
    weight_vectors_per_task = model_ltl.fit_inner(test_tasks_training_features,
                                                  all_labels=test_tasks_training_labels,
                                                  extra_inputs=extra_inputs)
    """
    Take the weight vectors from the previous step and make predictions on the test data of the same tasks you optimized them on
    """

    predictions_test = model_ltl.predict(test_tasks_test_features, weight_vectors_per_task, extra_inputs=extra_inputs)

    ltl_test_performance = multiple_tasks_mse(test_tasks_test_labels, predictions_test, extra_inputs['predictions_for_each_training_task'])
    print(ltl_test_performance)

    ###################################################################################################
    from src.independent_learning import ITL

    extra_inputs = {'point_indexes_per_task': point_indexes_per_test_task}

    model_itl = ITL(regularization_parameter=1e-6)
    model_itl.fit(test_tasks_training_features, test_tasks_training_labels, extra_inputs=extra_inputs)

    predictions_validation = model_itl.predict(test_tasks_test_features, extra_inputs=extra_inputs)

    val_performance = multiple_tasks_mse(test_tasks_test_labels, predictions_validation)
    print(val_performance)

    predictions_test = model_itl.predict(test_tasks_test_features, extra_inputs=extra_inputs)

    itl_test_performance = multiple_tasks_mse(test_tasks_test_labels, predictions_test)
    print(itl_test_performance)

    ###################################################################################################

    import matplotlib.pyplot as plt
    plt.plot(ltl_test_performance, color='tab:red', label='BiasLTL')
    plt.axhline(y=itl_test_performance, xmin=0, xmax=len(ltl_test_performance)-1, color='tab:blue', label='ITL')
    plt.show()



    ###########
    training_settings_itl = {'regularization_parameter_range': [10 ** float(i) for i in np.linspace(-12, 4, 20)],
                             'method': 'ITL'}

    results_itl = training(data, training_settings_itl)
    ###########

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
