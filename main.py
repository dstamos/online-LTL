from src.training import *
from src.data_management import DataHandler


def main():
    pass
    # training(data, data_settings, training_settings)


if __name__ == "__main__":

    seed = 999
    training_tasks_pct = 0.5
    validation_tasks_pct = 0.05
    test_tasks_pct = 0.45

    training_points_pct = 0.5
    validation_points_pct = 0.25
    test_points_pct = 0.25

    np.random.randn(seed)
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
    # Aist of all the features and labels for all tasks basically
    n_tasks = 100
    dims = 30
    noise = 1
    all_features = []
    all_labels = []
    common_mean = 2 * np.random.randn(dims)
    for task_idx in range(n_tasks):
        # Total number of points for the current task.
        n_points = np.random.randint(low=60, high=100)

        # Generating and normalizing the data.
        features = random.randn(n_points, dims)
        features = features / norm(features, axis=1, keepdims=True)

        # Generating the weight vector "around" the common mean.
        weight_vector = common_mean + np.random.randn(dims)

        # Linear model plus some noise.
        labels = features @ weight_vector + noise * random.randn(n_points)

        # Throwing the features and labels in their corresponding dictionary place.
        all_features.append(features)
        all_labels.append(labels)
    ########################################################################
    ########################################################################
    ########################################################################

    data = DataHandler(data_settings, all_features, all_labels)



    ##
    def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    if len(sys.argv) > 1:
        seed = int(sys.argv[1])

        n_points = int(sys.argv[2])
        n_tasks = int(sys.argv[3])
        n_dims = int(sys.argv[4])
        method_IDX = int(sys.argv[5]) # 0 batch, 1 online
        dataset_IDX = int(sys.argv[6])
        if method_IDX == 1:
            c_value = int(sys.argv[7])
        else:
            c_value = np.nan
            lambda_idx = int(sys.argv[7])
    else:
        seed = 999
        n_points = 110
        n_tasks = 75
        n_dims = 50
        method_IDX = 2 # 0 batch, 1 online, 2 MTL, 3 ITL
        dataset_IDX = 0
        if method_IDX == 1:
            c_value = 10**6
        else:
            c_value = np.nan
            lambda_idx = 999 # 14

    # lambda_range = [10 ** float(i) for i in np.linspace(-6, 5, 25)]
    lambda_range = [10 ** float(i) for i in np.linspace(-2, 5, 25)]

    data_settings = {}
    data_settings['seed'] = seed
    data_settings['n_points'] = n_points
    data_settings['n_dims'] = n_dims
    data_settings['n_tasks'] = n_tasks

    np.random.seed(data_settings['seed'])

    if dataset_IDX == 0:
        pass
    elif dataset_IDX == 1:
        n_train_tasks = n_tasks
        n_val_tasks = 100 - n_tasks
        n_test_tasks = 39
        n_tasks = n_train_tasks+n_val_tasks+n_test_tasks

        task_shuffled = np.random.permutation(n_tasks)

        data_settings['task_range_tr'] = task_shuffled[0:n_train_tasks]
        data_settings['task_range_val'] = task_shuffled[n_train_tasks:n_train_tasks+n_val_tasks]
        data_settings['task_range_test'] = task_shuffled[n_train_tasks+n_val_tasks:]

        data_settings['task_range'] = task_shuffled
        data_settings['n_tasks'] = n_tasks


    training_settings = {}
    if method_IDX == 1:
        nuclear_option = 0
        training_settings['param1_range'] = lambda_range
    else:
        if lambda_idx == 999:
            nuclear_option = 1
        else:
            nuclear_option = 0
            training_settings['param1_range'] = [lambda_range[lambda_idx]]



    # setting for step size on online LTL
    training_settings['c_value'] = c_value
    # setting for non-online methods
    training_settings['conv_tol'] = 10 ** -5

    if nuclear_option == 1:
        for idx, _ in enumerate(lambda_range):
            training_settings['param1_range'] = [lambda_range[idx]]
            print('working on lambda: %20.15f' % training_settings['param1_range'][0])

            if method_IDX == 0:
                training_settings['method'] = 'batch_LTL'
                training_settings['filename'] = "seed_" + str(seed) + '-lambda_' + str(
                    training_settings['param1_range'][0])
            elif method_IDX == 1:
                training_settings['method'] = 'online_LTL'
                training_settings['filename'] = "seed_" + str(seed) + '-c_value_' + str(c_value)
            elif method_IDX == 2:
                training_settings['method'] = 'MTL'
                training_settings['filename'] = "seed_" + str(seed) + '-lambda_' + str(training_settings['param1_range'][0])
            elif method_IDX == 3:
                training_settings['method'] = 'Validation_ITL'
                training_settings['filename'] = "seed_" + str(seed) + '-lambda_' + str(training_settings['param1_range'][0])

            if dataset_IDX == 0:
                data_settings['dataset'] = 'synthetic_regression'

                training_settings['foldername'] = 'results/' + data_settings['dataset'] + '-T_' + \
                                                  str(n_tasks) + '-n_' + str(n_points) + '/' \
                                                  + training_settings['method']
            elif dataset_IDX == 1:
                data_settings['dataset'] = 'schools'
                training_settings['foldername'] = 'results/' + data_settings['dataset'] + '-T_' + \
                                                  str(n_tasks) + '/' + training_settings['method']

            main(data_settings, training_settings)
    else:

        if method_IDX == 0:
            training_settings['method'] = 'batch_LTL'
            training_settings['filename'] = "seed_" + str(seed) + '-lambda_' + str(training_settings['param1_range'][0])
        elif method_IDX == 1:
            training_settings['method'] = 'online_LTL'
            training_settings['filename'] = "seed_" + str(seed) + '-c_value_' + str(c_value)
        elif method_IDX == 2:
            training_settings['method'] = 'MTL'
            training_settings['filename'] = "seed_" + str(seed) + '-lambda_' + str(training_settings['param1_range'][0])
        elif method_IDX == 3:
            training_settings['method'] = 'Validation_ITL'
            training_settings['filename'] = "seed_" + str(seed) + '-lambda_' + str(training_settings['param1_range'][0])

        if dataset_IDX == 0:
            data_settings['dataset'] = 'synthetic_regression'

            training_settings['foldername'] = 'results/' + data_settings['dataset'] + '-T_' + \
                                              str(n_train_tasks) + '-n_' + str(n_points) + '/' \
                                              + training_settings['method']
        elif dataset_IDX == 1:
            data_settings['dataset'] = 'schools'
            training_settings['foldername'] = 'results/' + data_settings['dataset'] + '-T_' + \
                                              str(n_train_tasks) + '/' + training_settings['method']
        main(data_settings, training_settings)




    print("done")
