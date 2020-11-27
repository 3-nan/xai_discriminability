import os
import numpy as np
import pandas as pd
import tensorflow as tf


def preprocess_imagenet(image, label):
    image = tf.io.read_file(image)

    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])

    image = tf.keras.applications.vgg16.preprocess_input(image)

    return image, tf.one_hot(label, depth=1000)


class Dataloader:

    def __init__(self, datapath, batch_size):
        self.datapath = datapath
        self.batch_size = batch_size

    def get_data(self, partition):
        """ Get the dataset with labels. """

        path = self.datapath + partition

        label_dict = pd.read_csv(self.datapath + "imagenet1000_clsid_to_labels.txt", delimiter=":", header=None)

        image_paths = []
        labels = []

        for root, dirs, files in os.walk(path):
            print(root)
            for filename in files:

                label_str = root.split("/")[-1]

                label = label_dict[label_dict[0] == label_str].index[0]

                image_paths.append(root + "/" + filename)
                labels.append(label)

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

        dataset = dataset.map(preprocess_imagenet)
        dataset = dataset.batch(batch_size=self.batch_size)

        return dataset

    def get_data_partition(self, partition, start_index, end_index):
        """ Get a partition of the dataset starting from start_index to end_index"""

        dataset = self.get_data(partition)

        dataset = dataset.skip(int(start_index / self.batch_size))
        dataset = dataset.take(int(end_index / self.batch_size) - int(start_index / self.batch_size))
        return dataset

    def get_indices(self, partition):
        """ Get the indices of the dataset. """
        dataset = self.get_data(partition)
        indices = []
        for batch in dataset.as_numpy_iterator():
            for index in batch[1]:
                indices.append(index)

        indices = np.array(indices)
        print(indices.shape)
        classes = np.where(np.sum(indices, axis=0) > 0)[0]
        return indices, classes

    def get_bounding_boxes(self):
        """ Get the bounding boxes."""
        pass
