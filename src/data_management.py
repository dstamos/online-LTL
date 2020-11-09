from sklearn.model_selection import train_test_split
from collections import namedtuple


class DataHandler:
    def __init__(self, settings, all_features, all_labels):
        """
        settings dict:
            settings = {'training_tasks_pct': training_tasks_pct,
                        'validation_tasks_pct': validation_tasks_pct,
                        'test_tasks_pct': test_tasks_pct,
                        'training_points_pct': training_points_pct,
                        'validation_points_pct': validation_points_pct,
                        'test_points_pct': test_points_pct}
        all_features:
            a list of X matrices of length n_tasks (each X matrix: (n_points, dims))
        all_labels:
            a list of y vectors of length n_tasks (each y vector: (n_points,))
        """
        self.settings = settings
        self.dims = all_features[0].shape[1]

        # Split the tasks _indexes_ into training/validation/test
        training_tasks_pct = settings['training_tasks_pct']
        validation_tasks_pct = settings['validation_tasks_pct']
        test_tasks_pct = settings['test_tasks_pct']
        training_tasks_indexes, temp_indexes = train_test_split(range(len(all_features)), test_size=1 - training_tasks_pct, shuffle=True)
        validation_tasks_indexes, test_tasks_indexes = train_test_split(temp_indexes, test_size=test_tasks_pct / (test_tasks_pct + validation_tasks_pct))

        self.training_tasks_indexes = training_tasks_indexes
        self.validation_tasks_indexes = validation_tasks_indexes
        self.test_tasks_indexes = test_tasks_indexes

        self.training_tasks = None
        self.validation_tasks = None
        self.test_tasks = None

        self._data_management(all_features, all_labels)

    def _data_management(self, all_features, all_labels):
        """
        The goal here to create 3 lists for the training/validation/test tasks.
        Each task needs to be split into training/validation/test datasets.
        Examples of output:

        data.training_tasks[linear_task_index].training.features
        data.test_tasks[linear_task_index].validation.labels
        """
        training_points_pct = self.settings['training_points_pct']
        validation_points_pct = self.settings['validation_points_pct']
        test_points_pct = self.settings['test_points_pct']

        def dataset_splits(task_indexes):
            bucket = []
            for task_index in task_indexes:
                # Split the dataset for the current tasks into training/validation/test
                training_features, temp_features, training_labels, temp_labels = train_test_split(all_features[task_index], all_labels[task_index], test_size=1 - training_points_pct, shuffle=True)
                validation_features, test_features, validation_labels, test_labels = train_test_split(temp_features, temp_labels, test_size=test_points_pct / (test_points_pct + validation_points_pct))

                training = namedtuple('Data', ['n_points', 'features', 'labels'])
                training.features = training_features
                training.labels = training_labels
                training.n_points = training_features.shape[0]

                validation = namedtuple('Data', ['n_points', 'features', 'labels'])
                validation.features = validation_features
                validation.labels = validation_labels
                validation.n_points = validation_features.shape[0]

                test = namedtuple('Data', ['n_points', 'features', 'labels'])
                test.features = test_features
                test.labels = test_labels
                test.n_points = test_features.shape[0]

                SetType = namedtuple('SetType', ['training', 'validation', 'test', 'n_tasks'])
                data = SetType(training, validation, test, len(task_indexes))

                bucket.append(data)
            return bucket

        self.training_tasks = dataset_splits(self.training_tasks_indexes)
        self.validation_tasks = dataset_splits(self.validation_tasks_indexes)
        self.test_tasks = dataset_splits(self.test_tasks_indexes)
