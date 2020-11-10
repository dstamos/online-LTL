from src.independent_learning import itl


def training(data, training_settings):
    method = training_settings['method']

    if method == 'ITL':
        results = itl(data, training_settings)

    elif method == 'online_LTL':
        pass

    return results
