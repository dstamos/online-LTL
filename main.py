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

    print('ITL')
    data = DataHandler(data_settings, all_features, all_labels)

    training_settings_itl = {'regularization_parameter_range': [10 ** float(i) for i in np.linspace(-12, 4, 100)],
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
    training_settings_online_ltl = {'regularization_parameter_range': [10 ** float(i) for i in np.linspace(-12, 4, 30)],
                                    'step_size': 1e+3,
                                    'method': 'online_LTL'}

    results_online_ltl = training(data, training_settings_online_ltl)

    import matplotlib.pyplot as plt
    plt.axhline(y=results_itl['test_perfomance'], xmin=0, xmax=len(all_features)-1, color='k', label='indipendent learning')
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
