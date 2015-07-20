from neon.datasets.dataset import Dataset
import logging
import numpy as np
import scipy.io
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class Matlab(Dataset):
    """
    Sets up a Matlab dataset.
    """

    def __init__(self, data_file, inputs_name = 'features', targets_name = 'targets', 
            train_size = None, test_size = None, normalize_mean = True, normalize_std = True, **kwargs):
        self.__dict__.update(kwargs)
        self.data_file = data_file
        self.inputs_name = inputs_name
        self.targets_name = targets_name
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

        # split data into training and test set
        (self.inputs['train'], self.inputs['test'], self.targets['train'], self.targets['test']) = \
            train_test_split(inputs, targets, train_size = self.train_size, test_size = self.test_size)
        logger.info('training set size: %d, test set size: %d', self.inputs['train'].shape[0], self.inputs['test'].shape[0])
        
        # normalize to zero mean and unit variance
        scaler = StandardScaler(with_mean = self.normalize_mean, with_std = self.normalize_std, copy = False)
        self.inputs['train'] = scaler.fit_transform(self.inputs['train'])
        self.inputs['test'] = scaler.transform(self.inputs['test'])

        self.format()
