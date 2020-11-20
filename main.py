import numpy as np
from numpy.linalg.linalg import norm
from src.ltl import BiasLTL
from src.utilities import multiple_tasks_mse
from src.data_management import split_data
from src.data_management import concatenate_data


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

    # Split the data into training/validation/test tasks.
    data = split_data(all_features, all_labels, data_settings)

    # Training
    # TODO Wrap the validation from this point
    model_ltl = BiasLTL(regularization_parameter=1e-1, step_size_bit=1e+3)
    training_tasks_training_features, training_tasks_training_labels, point_indexes_per_training_task = concatenate_data(data['training_tasks_training_features'], data['training_tasks_training_labels'])
    model_ltl.fit(training_tasks_training_features, training_tasks_training_labels, {'point_indexes_per_task': point_indexes_per_training_task})

    # Validation
    validation_tasks_training_features, validation_tasks_training_labels, point_indexes_per_validation_task = concatenate_data(data['validation_tasks_training_features'], data['validation_tasks_training_labels'])
    extra_inputs = {'predictions_for_each_training_task': False, 'point_indexes_per_task': point_indexes_per_validation_task}
    weight_vectors_per_task = model_ltl.fit_inner(validation_tasks_training_features, all_labels=validation_tasks_training_labels, extra_inputs=extra_inputs)

    validation_tasks_test_features, _, point_indexes_per_validation_task = concatenate_data(data['validation_tasks_test_features'], data['validation_tasks_test_labels'])
    extra_inputs['point_indexes_per_task'] = point_indexes_per_validation_task
    predictions_validation = model_ltl.predict(validation_tasks_test_features, weight_vectors_per_task, extra_inputs=extra_inputs)

    val_performance = multiple_tasks_mse(data['validation_tasks_test_labels'], predictions_validation, extra_inputs['predictions_for_each_training_task'])
    print(val_performance)
    # TODO (to this point) Use val_performance to pick the best regularization_parameter. If predictions_for_each_training_task = True, then val_performance is a list, not a scalar.

    # Test
    test_tasks_training_features, test_tasks_training_labels, point_indexes_per_test_task = concatenate_data(data['test_tasks_training_features'], data['test_tasks_training_labels'])
    extra_inputs = {'predictions_for_each_training_task': True, 'point_indexes_per_task': point_indexes_per_test_task}
    weight_vectors_per_task = model_ltl.fit_inner(test_tasks_training_features, all_labels=test_tasks_training_labels, extra_inputs=extra_inputs)

    test_tasks_test_features, _, point_indexes_per_test_task = concatenate_data(data['test_tasks_test_features'], data['test_tasks_test_labels'])
    extra_inputs['point_indexes_per_task'] = point_indexes_per_test_task
    predictions_test = model_ltl.predict(test_tasks_test_features, weight_vectors_per_task, extra_inputs=extra_inputs)

    ltl_test_performance = multiple_tasks_mse(data['test_tasks_test_labels'], predictions_test, extra_inputs['predictions_for_each_training_task'])
    print(ltl_test_performance)

    ###################################################################################################
    ###################################################################################################
    ###################################################################################################
    # Independent learning on the test tasks.
    from src.independent_learning import ITL
    test_tasks_training_features, test_tasks_training_labels, point_indexes_per_test_task = concatenate_data(data['test_tasks_training_features'], data['test_tasks_training_labels'])
    extra_inputs = {'point_indexes_per_task': point_indexes_per_test_task}

    model_itl = ITL(regularization_parameter=1e-6)
    model_itl.fit(test_tasks_training_features, test_tasks_training_labels, extra_inputs=extra_inputs)

    test_tasks_validation_features, _, point_indexes_per_test_task = concatenate_data(data['test_tasks_validation_features'], data['test_tasks_validation_labels'])
    extra_inputs = {'point_indexes_per_task': point_indexes_per_test_task}
    predictions_validation = model_itl.predict(test_tasks_validation_features, extra_inputs=extra_inputs)

    val_performance = multiple_tasks_mse(data['test_tasks_validation_labels'], predictions_validation)
    print(val_performance)
    # TODO Use val_performance to pick the best regularization_parameter.

    test_tasks_test_features, _, point_indexes_per_test_task = concatenate_data(data['test_tasks_test_features'], data['test_tasks_test_labels'])
    extra_inputs = {'point_indexes_per_task': point_indexes_per_test_task}
    predictions_test = model_itl.predict(test_tasks_test_features, extra_inputs=extra_inputs)

    itl_test_performance = multiple_tasks_mse(data['test_tasks_test_labels'], predictions_test)
    print(itl_test_performance)
    ###################################################################################################
    ###################################################################################################
    ###################################################################################################

    import matplotlib.pyplot as plt
    plt.plot(ltl_test_performance, color='tab:red', label='BiasLTL')
    plt.axhline(y=itl_test_performance, xmin=0, xmax=len(ltl_test_performance)-1, color='tab:blue', label='ITL')
    plt.show()

    print('done')


if __name__ == "__main__":
    main()
