import numpy as np
from numpy.linalg.linalg import norm
from src.ltl import BiasLTL
from src.utilities import multiple_tasks_mae_clip
from src.data_management import split_data
from src.data_management import concatenate_data

from sklearn.preprocessing import StandardScaler
from src.preprocessing import ThressholdScaler
from src.loadData import load_data_essex, split_data_essex

import pickle

def single_fold(data, regularization, each_training):
    # Training
    model_ltl = BiasLTL(regularization_parameter=regularization, step_size_bit=1e+3)
    outlier = ThressholdScaler()
    norm = StandardScaler()
    training_features, training_labels, point_indexes_per_task = concatenate_data(
        data['training_tasks_training_features'], data['training_tasks_training_labels'])
    training_features = outlier.fit_transform(training_features)
    training_features = norm.fit_transform(training_features)
    training_features = np.concatenate((np.ones((len(training_features), 1)), training_features), 1)
    model_ltl.fit(training_features, training_labels, {'point_indexes_per_task': point_indexes_per_task})

    # Testing. This should be kept fairly similar.
    if len(data['validation_tasks_training_features']):
        re_train_features, re_train_labels, re_train_indexes = concatenate_data(data['validation_tasks_training_features'],
                                                                                data['validation_tasks_training_labels'])
        re_train_features = outlier.transform(re_train_features)
        re_train_features = norm.transform(re_train_features)
        re_train_features = np.concatenate((np.ones((len(re_train_features), 1)), re_train_features), 1)
    else:
        re_train_features = []
        re_train_labels = []
        re_train_indexes = []
    extra_inputs = {'predictions_for_each_training_task': each_training,
                    're_train_indexes': re_train_indexes,
                    're_train_features': re_train_features,
                    're_train_labels': re_train_labels
                    }
    test_features, test_labels, point_indexes_per_test_task = concatenate_data(
        data['test_tasks_training_features'], data['test_tasks_training_labels'])
    test_features = outlier.transform(test_features)
    test_features = norm.transform(test_features)
    test_features = np.concatenate((np.ones((len(test_features), 1)), test_features), 1)

    extra_inputs['point_indexes_per_task'] = point_indexes_per_test_task
    predictions_test = model_ltl.predict(test_features, extra_inputs=extra_inputs)
    perf = multiple_tasks_mae_clip(data['test_tasks_training_labels'], predictions_test,
                                             extra_inputs['predictions_for_each_training_task'])
    return predictions_test, perf, (outlier, norm, model_ltl)


def gridSearch(feats, labels, test, reg_param, ret):
    nSubj = int(len(all_features) / 3)
    ind = np.arange(nSubj * 3, dtype=int)
    test_i = np.arange(3, dtype=int) + test*3
    bestPerf = 1
    performance = []
    param_perf = np.zeros(len(reg_param))
    for i, param in enumerate(reg_param):#Search for each possible regurlaization paramether
        for s in range(nSubj): #In a cross validation manner
            if s != test:
                val = np.arange(3, dtype=int) + s*3
                train = np.setdiff1d(ind, test_i)
                train = np.setdiff1d(train, val)
                data = split_data_essex(feats, labels, train, val[:ret], val[ret:])
                foo, perf, foo = single_fold(data, param, False)
                performance.append(perf)
        param_perf[i] = np.mean(performance)
        if param_perf[i] < bestPerf: #Save the best one
            bestPerf = param_perf[i]
            best_param = param
    train = np.setdiff1d(ind, test_i)
    data = split_data_essex(feats, labels, train, test_i[:ret], test_i[ret:])  # Retrain with train + val with the best
    prediction, perf, mdl = single_fold(data, best_param, True)
    return param_perf, prediction, perf, mdl



def recalculate_perf(all_labels, prediction, days=3):
    nSubj = len(prediction)
    error = np.zeros(nSubj)
    for s in range(nSubj):
        p = np.clip(prediction[s], 0.1, 1)
        test = np.arange(3, dtype=int) + s * 3
        val = test[3-days:]
        y = []
        for i in val:
            y = np.concatenate((y, all_labels[i]))
        error[s] = np.median(np.abs(y-p))
    return error

if __name__ == "__main__":

    all_features, all_labels = load_data_essex()

    nSubj = int(len(all_features) / 3)

    prediction = []
    models = []
    bestmp = []
    performance = np.zeros((nSubj, 24))
    reg_param = [1e-3, 1e-2, 1e-1, 1, 10]
    for s in range(nSubj): # Leave one subject out retrain 1 day
        pp, pred, perf, mdl = gridSearch(all_features, all_labels, s, reg_param, 1)
        performance[s] = perf
        prediction.append(pred)
        models.append(mdl)
        bestmp.append(pp)
    np.save('performance_1_retrain.npy', performance)
    with open('results_1_retrain.pkl', 'wb') as f:  
        pickle.dump([prediction, models, bestmp], f)

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
