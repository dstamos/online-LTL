import numpy as np
from numpy.linalg.linalg import norm
from src.ltl import BiasLTL
from src.utilities import multiple_tasks_mse
from src.data_management import split_data
from src.data_management import concatenate_data

from sklearn.preprocessing import StandardScaler
from src.preprocessing import ThressholdScaler
from src.loadData import load_data_essex, split_data_essex

if __name__ == "__main__":

    all_features, all_labels = load_data_essex()

    # Split the data into training/validation/test tasks.
    data = split_data_essex(all_features, all_labels, [0, 1], [2])
    # Training
    # TODO Wrap the validation from this point: Create a loop with the possible regularization parameters. Each loop
    #  calls to train_test_split using the training data. The best model is selected before the retrain and test.
    model_ltl = BiasLTL(regularization_parameter=1e-1, step_size_bit=1e+3)
    outlier = ThressholdScaler()
    norm = StandardScaler()
    training_features, training_labels, point_indexes_per_task = concatenate_data(data['training_tasks_training_features'], data['training_tasks_training_labels'])
    training_features = outlier.fit_transform(training_features)
    training_features = norm.fit_transform(training_features)
    model_ltl.fit(training_features, training_labels, {'point_indexes_per_task': point_indexes_per_task})


    # Testing. This should be kept fairly similar.
    re_train_features, re_train_labels, re_train_indexes = concatenate_data(data['validation_tasks_training_features'], data['validation_tasks_training_labels'])
    re_train_features = outlier.transform(re_train_features)
    re_train_features = norm.transform(re_train_features)
    extra_inputs = {'predictions_for_each_training_task': False,
                    're_train_indexes': re_train_indexes,
                    're_train_features': re_train_features,
                    're_train_labels': re_train_labels
                    }
    test_features, test_labels, point_indexes_per_test_task = concatenate_data(
        data['test_tasks_training_features'], data['test_tasks_training_labels'])
    test_features = outlier.transform(test_features)
    test_features = norm.transform(test_features)

    extra_inputs['point_indexes_per_task'] = point_indexes_per_test_task
    predictions_test = model_ltl.predict(test_features, extra_inputs=extra_inputs)
    ltl_test_performance = multiple_tasks_mse(data['test_tasks_training_labels'], predictions_test,
                                              extra_inputs['predictions_for_each_training_task'])
    print(ltl_test_performance)

    if False:
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