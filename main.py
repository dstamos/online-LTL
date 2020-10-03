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
    dims = 30
    noise = 0.5
    all_features = []
    all_labels = []
    common_mean = 5 * np.random.randn(dims)
    for task_idx in range(n_tasks):
        # Total number of points for the current task.
        n_points = np.random.randint(low=40, high=50)

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

    data = DataHandler(data_settings, all_features, all_labels)

    # training_settings = {'regularization_parameter_range': [10 ** float(i) for i in np.linspace(-12, 4, 36)],
    #                      'method': 'ITL'}

    # training_settings = {'regularization_parameter': 1e-2,
    #                      'method': 'batch_LTL'}
    #
    training_settings = {'regularization_parameter_range': [10 ** float(i) for i in np.linspace(-2, 0.3, 8)],
                         'step_size': 1e+3,
                         'method': 'online_LTL'}

    # TODO What is the exact interaction that breaks online ltl when the regul param is large?
    # TODO Finish online ltl
    #
    # training_settings = {'regularization_parameter': 1e-2,
    #                      'method': 'MTL'}

    training(data, training_settings)


if __name__ == "__main__":
    main()
