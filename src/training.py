from src.independent_learning import itl
from src.ltl import variance_online_ltl


def training(data, training_settings):
    method = training_settings['method']

    if method == 'ITL':
        itl(data, training_settings)

    elif method == 'batch_LTL':
        pass
        # batch_ltl(data, data_settings, training_settings)

    elif method == 'online_LTL':
        variance_online_ltl(data, training_settings)

    print('done')
    return
