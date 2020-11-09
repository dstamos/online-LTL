from src.independent_learning import itl
from src.ltl import variance_online_ltl


def training(data, training_settings):
    method = training_settings['method']

    if method == 'ITL':
        results = itl(data, training_settings)

    elif method == 'online_LTL':
        results = variance_online_ltl(data, training_settings)

    return results
