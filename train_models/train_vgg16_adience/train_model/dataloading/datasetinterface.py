import collections
import xml
import xmltodict
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod


class DataSample(ABC):
    """ Represents a single sample of a dataset. """

    def __init__(self, image, filename):
        """ Initializes a new DataSample instance. """
        self.image = image
        self.filename = filename


class Dataset(ABC):

    def __init__(self, datapath, partition):
        """ Initialize the model. """
        self.datapath = datapath

        assert partition in ["train", "val"]
        self.partition = partition
        self.samples = []
        self.mode = "raw"
        super().__init__()

    def __len__(self):
        """ Returns the length of the dataset. """
        return len(self.samples)

    @abstractmethod
    def __getitem__(self, index):
        """ Retrieves the element with the specified index. """
        return NotImplementedError

    def set_mode(self, mode):
        """ Set the mode of the dataset to determine return values"""
        if mode not in ["raw", "preprocessed", "binary_mask"]:
            raise ValueError(f"mode {mode} not in the set of valid options")

        self.mode = mode

    # @abstractmethod
    # def get_dataset_partition(self, startidx, endidx, batched=False):
    #     """ Retrieve a partition from the dataset ranging from startidx to endidx. """
    #     return NotImplementedError
    #
    # @abstractmethod
    # def preprocess_image(self, image):
    #     """ Preprocess a single image. """
    #     return NotImplementedError
    #
    # @abstractmethod
    # def preprocess_data(self, data, labels):
    #     """ Preprocess the presented data. """
    #     preprocessed = [self.preprocess_image(image, label) for image, label in np.column_stack(data, labels)]
    #
    #     return preprocessed[:, 0], preprocessed[:, 1]
