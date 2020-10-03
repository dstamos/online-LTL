from src.independent_learning import itl
from src.ltl import variance_online_ltl, variance_batch_ltl


def training(data, training_settings):
    method = training_settings['method']

    if method == 'ITL':
        itl(data, training_settings)

    elif method == 'batch_LTL':
        variance_batch_ltl(data, training_settings)

    elif method == 'online_LTL':
        variance_online_ltl(data, training_settings)

    return
