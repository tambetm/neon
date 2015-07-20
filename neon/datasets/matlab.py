from neon.datasets.dataset import Dataset
import logging
import numpy as np
import scipy.io

logger = logging.getLogger(__name__)

class Matlab(Dataset):
    """
    Sets up a Matlab dataset.
    """

    def __init__(self, data_file, inputs_name = 'features', targets_name = 'targets', shuffle = True,
            train_size = None, test_size = None, normalize_mean = True, normalize_std = True, **kwargs):
        self.__dict__.update(kwargs)
        self.data_file = data_file
        self.inputs_name = inputs_name
        self.targets_name = targets_name
        self.shuffle = shuffle
        self.train_size = train_size
        self.test_size = test_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    def load(self, backend=None, experiment=None):
        # don't try to load data again if already loaded
        if self.inputs['train'] is not None:
            return

        # load Matlab file and extract matrices
        logger.info('loading: %s', self.data_file)
        data = scipy.io.loadmat(self.data_file)
        inputs = data[self.inputs_name].astype(float)
        targets = data[self.targets_name]
        logger.info('inputs: %s, targets: %s', str(inputs.shape), str(targets.shape))
        assert inputs.shape[0] == targets.shape[0]
        nr_samples = inputs.shape[0]

        if self.shuffle:
            indices = np.random.permutation(nr_samples)
        else:
            indices = range(nr_samples)

        # split data into training and test set
        if self.train_size is not None and isinstance(self.train_size, float):
            assert 0 <= self.train_size <= 1
            self.train_size = int(self.train_size * nr_samples)
        if self.test_size is not None and isinstance(self.test_size, float):
            assert 0 <= self.test_size <= 1
            self.test_size = int(self.test_size * nr_samples)

        if self.train_size is not None and self.test_size is not None:
            self.train_size = nr_samples - self.test_size
        elif self.test_size is None and self.train_size is not None:
            self.test_size = nr_samples - self.train_size

        assert 0 <= self.train_size <= nr_samples
        assert 0 <= self.test_size <= nr_samples
        assert (self.train_size + self.test_size) <= nr_samples

        self.inputs['train'] = inputs[indices[self.test_size:(self.train_size + self.test_size)], ...]
        self.inputs['test'] = inputs[indices[:self.test_size], ...]
        self.targets['train'] = targets[indices[self.test_size:(self.train_size + self.test_size)], ...]
        self.targets['test'] = targets[indices[:self.test_size], ...]
        logger.info('training set size: %d, test set size: %d', self.inputs['train'].shape[0], self.inputs['test'].shape[0])
        
        # normalize to zero mean and unit variance
        mean = np.mean(self.inputs['train'], axis=0)
        std = np.std(self.inputs['train'], axis=0)
        assert len(mean.shape) == 1 and mean.shape[0] == inputs.shape[1]
        assert len(std.shape) == 1 and std.shape[0] == inputs.shape[1]
        std[std == 0.0] = 1.0

        self.inputs['train'] -= mean
        self.inputs['train'] /= std
        self.inputs['test'] -= mean
        self.inputs['test'] /= std

        self.format()
