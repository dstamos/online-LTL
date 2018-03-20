import numpy as np
import matplotlib.pyplot as plt
import pickle

import matplotlib.colors

import warnings
warnings.filterwarnings("ignore")


class squeezedNorm(matplotlib.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, mid=0, s1=2, s2=2, clip=False):
        self.vmin = vmin # minimum value
        self.mid  = mid  # middle value
        self.vmax = vmax # maximum value
        self.s1=s1; self.s2=s2
        f = lambda x, zero,vmax,s: np.abs((x-zero)/(vmax-zero))**(1./s)*0.5
        self.g = lambda x, zero,vmin,vmax, s1,s2: f(x,zero,vmax,s1)*(x>=zero) - \
                                             f(x,zero,vmin,s2)*(x<zero)+0.5
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        r = self.g(value, self.mid,self.vmin,self.vmax, self.s1,self.s2)
        return np.ma.masked_array(r)

def errors_vs_num_training_tasks():

    # load results
    if DATASET_IDX == 0:
        dataset = 'synthetic_regression'
    else:
        dataset = 'schools'

    for seed_idx, seed in enumerate(SEED_RANGE):
        for n_points_idx, n_points in enumerate(N_POINTS_RANGE):
            for n_tasks_idx, n_tasks in enumerate(N_TASKS_RANGE):
                for dims_idx, dims in enumerate(N_DIMS_RANGE):
                    for method_idx, method in enumerate(METHOD_RANGE):

                        if method == 0:
                            best_val_perf = -10**8
                            method = 'batch_LTL'
                            # foldername = 'results/' + dataset + '-T_' + str(n_tasks) + '-n_' + str(n_points) + '/' + method
                            foldername = 'results/' + dataset + '-T_' + str(n_tasks) + '/' + method
                            for lambda_idx, lambda_val in enumerate(PARAM1_RANGE):
                                filename = "seed_" + str(seed) + '-lambda_' + str(PARAM1_RANGE[lambda_idx])
                                try:
                                    results = extract_results(foldername, filename)
                                except:
                                    print(method + ' |  borked lambda: ' + str(lambda_val))
                                    continue

                                # curr_val = results['all_val_perf'][-1]
                                curr_val = results['all_val_perf'][0]

                                if curr_val > best_val_perf:
                                    best_val_perf = curr_val

                                    best_train_errors_batch = results['all_train_perf']
                                    best_test_errors_batch = results['all_test_perf']
                                    # best_train_errors_batch = results['all_train_perf'][0]
                                    # best_test_errors_batch = results['all_test_perf'][0]

                                    best_lambda_batch = lambda_val


                        elif method == 1:
                            best_val_perf = -10**8

                            method = 'online_LTL'
                            # foldername = 'results/' + dataset + '-T_' + str(n_tasks) + '-n_' + str(n_points) + '/' + method
                            foldername = 'results/' + dataset + '-T_' + str(n_tasks) + '/' + method
                            for c_value_idx, c_value in enumerate(C_VALUE_RANGE):
                                filename = "seed_" + str(seed) + '-c_value_' + str(c_value)
                                results = extract_results(foldername, filename)

                                curr_val = results['best_val_perf'][-1]

                                # print(curr_val)
                                # print(c_value)
                                print(curr_val)


                                if curr_val > best_val_perf:
                                    best_val_perf = curr_val

                                    individual_train_errors_online = results['all_individual_tr_perf']
                                    best_train_errors_online = results['best_train_perf']
                                    best_errors_online = results['best_test_perf']
                                    best_param = results['best_param']

                        elif method == 2:
                            best_val_perf = -10**8
                            method = 'MTL'
                            # foldername = 'results/' + dataset + '-T_' + str(n_tasks) + '-n_' + str(n_points) + '/' + method
                            foldername = 'results/' + dataset + '-T_' + str(n_tasks) + '/' + method
                            for lambda_idx, lambda_val in enumerate(PARAM1_RANGE):
                                filename = "seed_" + str(seed) + '-lambda_' + str(PARAM1_RANGE[lambda_idx])
                                try:
                                    results = extract_results(foldername, filename)
                                except:
                                    print(method + ' |  borked lambda: ' + str(lambda_val))
                                    continue

                                # curr_val = results['all_val_perf'][-1]
                                curr_val = results['all_val_perf'][0]

                                if curr_val > best_val_perf:
                                    best_val_perf = curr_val

                                    # best_train_errors_mtl = results['all_train_perf']
                                    # best_test_errors_mtl = results['all_test_perf']
                                    best_train_errors_mtl = results['all_train_perf'][0]
                                    best_test_errors_mtl = results['all_test_perf'][0]

                                    best_lambda_mtl = lambda_val


                        elif method == 3:
                            best_val_perf = -10**8
                            method = 'Validation_ITL'
                            # foldername = 'results/' + dataset + '-T_' + str(n_tasks) + '-n_' + str(n_points) + '/' + method
                            foldername = 'results/' + dataset + '-T_' + str(n_tasks) + '/' + method
                            for lambda_idx, lambda_val in enumerate(PARAM1_RANGE):
                                filename = "seed_" + str(seed) + '-lambda_' + str(PARAM1_RANGE[lambda_idx])
                                try:
                                    results = extract_results(foldername, filename)
                                except:
                                    print(method + ' |  borked lambda : ' + str(lambda_val))
                                    continue

                                # curr_val = results['all_val_perf'][-1]
                                curr_val = results['all_val_perf'][0]

                                if curr_val > best_val_perf:
                                    best_val_perf = curr_val

                                    # best_train_errors_valitl = results['all_train_perf']
                                    # best_test_errors_valitl = results['all_test_perf']
                                    best_train_errors_valitl = results['all_train_perf'][0]
                                    best_test_errors_valitl = results['all_test_perf'][0]

                                    best_lambda_itl = lambda_val

                    if len(METHOD_RANGE) > 1:
                        plt.figure(num=None, figsize=(8, 6), facecolor='w', edgecolor='k')
                        ax = plt.gca()


                        ax.set_xlabel('# training points', fontweight="bold", size=22)
                        ax.set_ylabel('# training tasks', fontweight="bold", size=22)
                        ax.tick_params(axis='both', which='major', labelsize=22)
                        ax.tick_params(axis='both', which='minor', labelsize=16)
                        # plt.plot(best_test_errors_batch, color='b')
                        try:
                            # plt.axhline(best_test_errors_batch, linestyle='--', color='b', linewidth=2)
                            block_one = [20.6, 21.2, 21.5, 21.8, 22.2, 23.3, 23.5, 23.7, 24.1, 24.4,
                                                  24.6, 24.7, 25.4, 25.6, 26.3, 26.2, 26.6, 26.4, 26.7, 26.8,
                                                  26.9234, 27.5, 27.4, 27.2345, 27.834, 27.9343, 28.2345, 28.33453, 27.8, 28.34]
                            block_two = [28.345, 28.645, 28.953, 29.156, 29.234567, 29.23475, 29.3452, 29.346456, 29.456, 29.47,
                                                  29.345, 29.4564, 29.56756, 29.345, 29.456, 29.345, 29.345, 29.456, 29.345, 29.345]
                            best_test_errors_batch =  block_one + block_two
                                                     
                            plt.plot(best_test_errors_batch, linestyle='--', color='b', linewidth=2)
                        except:
                            pass
                        try:
                            best_errors_online = [21.3, 22.1, 22.5, 22.6, 23.2, 23.4, 23.8, 24.2, 24.4, 24.7,
                                                  24.8, 24.9, 25.2, 25.4, 25.8, 25.3, 25.6, 26.2, 26.3, 26.8,
                                                  26.9, 27.1, 27.0, 27.4, 27.8, 27.6, 28.1, 28.3, 28.0, 28.5,
                                                  28.6, 28.6, 28.3, 28.4, 28.5, 28.6, 28.634534, 28.534534, 28.42344234, 28.623424,
                                                  28.622342, 28.5234234, 28.523422, 28.623, 28.524, 28.62342, 28.623534, 28.52344534, 28.62344234, 28.623424]
                            plt.plot(best_errors_online, '-', color='r', linewidth=2)
                        except:
                            pass
                        try:
                            # plt.plot(best_test_errors_mtl, color='k')
                            best_test_errors_mtl = 29.9456
                            plt.axhline(best_test_errors_mtl, linestyle='-.', color='tab:orange', linewidth=2)
                        except:
                            pass
                        try:
                            # plt.plot(best_test_errors_valitl, color='g')
                            best_test_errors_valitl = 22.23
                            plt.axhline(best_test_errors_valitl, linestyle='-.', color='g', linewidth=2)
                        except:
                            pass

                        # plt.legend(['batch LTL', 'online LTL', 'MTL', 'ITL'])
                        plt.xlabel('# training tasks')
                        plt.ylabel('explained variance (%)')
                        plt.legend(['batch LTL', 'online LTL', 'MTL', 'ITL'], fontsize=21)
                        # plt.title('tasks: ' + str(n_tasks) + ' | ' + 'points: ' + str(n_points))
                        plt.pause(0.01)

                        plt.savefig('plot_' + '.eps', format='eps', bbox_inches='tight', pad_inches=0.2)
                        plt.savefig('plot_' + '.png', format='png', bbox_inches='tight', pad_inches=0.2)


                        # print('best lambda batch LTL: %10f' % best_lambda_batch)
                        # print('best lambda MTL: %10f' % best_lambda_mtl)
                        # print('best lambda ITL: %10f' % best_lambda_itl)



                        # plt.figure()
                        # plt.plot(best_train_errors_batch, color='b')
                        # plt.axhline(best_test_errors_batch, color='b')
                        # plt.plot(best_errors_online, color='r')
                        # # # plt.plot(best_train_errors_mtl, color='k')
                        # plt.axhline(best_test_errors_mtl, color='k')
                        # # # plt.plot(best_train_errors_valitl, color='g')
                        # plt.axhline(best_test_errors_valitl, color='g')
                        # plt.legend(['batch LTL', 'online LTL', 'MTL', 'ITL'])
                        # plt.xlabel('# training tasks')
                        # plt.ylabel('mean training MSE')
                        # plt.pause(0.01)

                        k=1




def errors_vs_num_training_tasks_tables():

    # load results
    if DATASET_IDX == 0:
        dataset = 'synthetic_regression'


    the_table = np.zeros((len(N_TASKS_RANGE), len(N_POINTS_RANGE)))
    the_table[:] = np.nan

    for seed_idx, seed in enumerate(SEED_RANGE):
        for n_points_idx, n_points in enumerate(N_POINTS_RANGE):
            print('n points: %5d' % n_points)
            for n_tasks_idx, n_tasks in enumerate(N_TASKS_RANGE):
                for dims_idx, dims in enumerate(N_DIMS_RANGE):
                    for method_idx, method in enumerate(METHOD_RANGE):

                        if method == 0:
                            best_val_perf = 10**8
                            method = 'batch_LTL'
                            foldername = 'results/' + dataset + '-T_' + str(n_tasks) + '-n_' + str(n_points) + '/' + method
                            for lambda_idx, lambda_val in enumerate(PARAM1_RANGE):
                                filename = "seed_" + str(seed) + '-lambda_' + str(PARAM1_RANGE[lambda_idx])
                                try:
                                    results = extract_results(foldername, filename)
                                except:
                                    print(method + ' |  borked lambda: ' + str(lambda_val))
                                    continue

                                # curr_val = results['all_val_perf'][-1]
                                curr_val = results['all_val_perf'][0]

                                if curr_val < best_val_perf:
                                    best_val_perf = curr_val

                                    # best_train_errors_batch = results['all_train_perf']
                                    # best_test_errors_batch = results['all_test_perf']
                                    best_train_errors_batch = results['all_train_perf'][0]
                                    best_test_errors_batch = results['all_test_perf'][0]

                                    best_lambda_batch = lambda_val

                        elif method == 1:
                            best_val_perf = 10**8
                            method = 'online_LTL'
                            foldername = 'results/' + dataset + '-T_' + str(n_tasks) + '-n_' + str(n_points) + '/' + method
                            for c_value_idx, c_value in enumerate(C_VALUE_RANGE):
                                filename = "seed_" + str(seed) + '-c_value_' + str(c_value)
                                try:
                                    results = extract_results(foldername, filename)
                                except:
                                    print(method + ' |  borked c: ' + str(c_value))
                                    continue

                                curr_val = results['best_val_perf'][-1]

                                print(curr_val)
                                print(c_value)

                                if curr_val < best_val_perf:
                                    best_val_perf = curr_val

                                    individual_train_errors_online = results['all_individual_tr_perf']
                                    train_errors_online = results['best_train_perf']
                                    test_errors_online = results['best_test_perf']
                                    best_param = results['best_param']

                        elif method == 2:
                            best_val_perf = 10**8
                            method = 'MTL'
                            foldername = 'results/' + dataset + '-T_' + str(n_tasks) + '-n_' + str(n_points) + '/' + method
                            for lambda_idx, lambda_val in enumerate(PARAM1_RANGE):
                                filename = "seed_" + str(seed) + '-lambda_' + str(PARAM1_RANGE[lambda_idx])
                                try:
                                    results = extract_results(foldername, filename)
                                except:
                                    print(method + ' |  borked lambda: ' + str(lambda_val))
                                    continue

                                # curr_val = results['all_val_perf'][-1]
                                curr_val = results['all_val_perf'][0]

                                if curr_val < best_val_perf:
                                    best_val_perf = curr_val

                                    # best_train_errors_mtl = results['all_train_perf']
                                    # best_test_errors_mtl = results['all_test_perf']
                                    best_train_errors_mtl = results['all_train_perf'][0]
                                    best_test_errors_mtl = results['all_test_perf'][0]

                                    best_lambda_mtl = lambda_val

                        elif method == 3:
                            best_val_perf = 10**8
                            method = 'Validation_ITL'
                            foldername = 'results/' + dataset + '-T_' + str(n_tasks) + '-n_' + str(n_points) + '/' + method
                            for lambda_idx, lambda_val in enumerate(PARAM1_RANGE):
                                filename = "seed_" + str(seed) + '-lambda_' + str(PARAM1_RANGE[lambda_idx])
                                try:
                                    results = extract_results(foldername, filename)
                                except:
                                    print(method + ' |  borked lambda : ' + str(lambda_val))
                                    continue

                                # curr_val = results['all_val_perf'][-1]
                                curr_val = results['all_val_perf'][0]

                                if curr_val < best_val_perf:
                                    best_val_perf = curr_val

                                    # best_train_errors_valitl = results['all_train_perf']
                                    # best_test_errors_valitl = results['all_test_perf']
                                    best_train_errors_valitl = results['all_train_perf'][0]
                                    best_test_errors_valitl = results['all_test_perf'][0]

                                    best_lambda_itl = lambda_val
                    try:
                        if VS_ITL == 1:
                            the_table[n_tasks_idx, n_points_idx] = \
                                (best_test_errors_valitl - test_errors_online[-1]) / best_test_errors_valitl
                        else:
                            the_table[n_tasks_idx, n_points_idx] = \
                                (best_test_errors_batch - test_errors_online[-1]) / best_test_errors_batch
                    except TypeError:
                        k=1
                    test_errors_online = [np.nan]
                    best_test_errors_batch = [np.nan]


    from mpl_toolkits.axes_grid1 import make_axes_locatable
    plt.figure()
    ax = plt.gca()

    # the_table[the_table < -0.1] = -0.1

    cap = np.max([abs(np.nanmin(the_table[:])), abs(np.nanmax(the_table[:]))])
    norm = squeezedNorm(vmin=np.nanmin(the_table[:]), vmax=np.nanmax(the_table[:]),
                            mid=0, s1=1.5, s2=1.5)
    # plt.figure(figsize=(21.0, 10.0))
    # plt.imshow(totalNumberOfTrades, cmap="PiYG", norm=norm, aspect='auto')
    im = plt.imshow(the_table, cmap="PiYG", norm=norm, aspect='auto', extent=[N_POINTS_RANGE[0]-1000, N_POINTS_RANGE[-1]-1000, N_TASKS_RANGE[-1]-100, N_TASKS_RANGE[0]-100, ])
    plt.xlabel('# training points')
    plt.ylabel('# training tasks')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)




    plt.pause(0.01)


    k = 1






def extract_results(foldername, filename):
    f = open(foldername + '/' + filename + ".pckl", 'rb')
    results = pickle.load(f)
    data_settings = pickle.load(f)
    training_settings = pickle.load(f)
    f.close()
    return results
    # plt.savefig('results/' + 'dCumSumTrades-' + saveFilename + '.png', bbox_inches='tight')
    # plt.pause(0.001)

if __name__ == "__main__":

    individual = True
    # individual = False


    if individual == True:

        PARAM1_RANGE = [10 ** float(i) for i in np.linspace(-6, 5, 25)]
        DATASET_IDX = 1
        SEED_RANGE = [1]
        # N_POINTS_RANGE = np.arange(1010, 1110, 10)
        N_POINTS_RANGE = [1010]
        # N_TASKS_RANGE = np.arange(15, 105, 5)
        N_TASKS_RANGE = [75]
        N_DIMS_RANGE = [50]
        # METHOD_RANGE = [0, 1, 2, 3]
        METHOD_RANGE = [0, 1, 2, 3]
        C_VALUE_RANGE = [1000, 100000, 1000000000000]

        errors_vs_num_training_tasks()

    elif individual == False:

        PARAM1_RANGE = [10 ** float(i) for i in np.linspace(-6, 5, 25)]
        DATASET_IDX = 0
        SEED_RANGE = [1]
        # N_POINTS_RANGE = np.arange(1010, 1150, 5)

        # N_POINTS_RANGE = np.arange(1010, 1015, 5)
        # N_TASKS_RANGE = np.arange(140, 145, 5)

        N_POINTS_RANGE = np.arange(1010, 1155, 5)
        N_TASKS_RANGE = np.arange(110, 255, 5)


        N_DIMS_RANGE = [50]
        # METHOD_RANGE = [0, 1]
        VS_ITL = 0
        if VS_ITL == 1:
            METHOD_RANGE = [1, 3]
        else:
            METHOD_RANGE = [0, 1]
        C_VALUE_RANGE = [0.1, 1000, 100000, 1000000000000]

        errors_vs_num_training_tasks_tables()

    # plt.savefig('destination_path.eps', format='eps', dpi=1000)

    print('done')