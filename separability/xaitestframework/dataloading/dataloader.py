import os
import sys
import collections
import xml
import xmltodict
import numpy as np
import pandas as pd
import tensorflow as tf
from abc import ABC, abstractmethod


def get_dataloader(classname):
    return getattr(sys.modules[__name__], classname)


class Dataloader(ABC):

    def __init__(self, datapath, partition, batch_size):
        """ Initialize the model. """
        self.datapath = datapath
        self.partition = partition
        self.batch_size = batch_size
        super().__init__()

    @abstractmethod
    def __len__(self):
        """ Returns the length of the dataset. """
        return NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        """ Retrieves the element with the specified index. """
        return NotImplementedError

    @abstractmethod
    def get_dataset_partition(self, startidx, endidx, batched=False):
        """ Retrieve a partition from the dataset ranging from startidx to endidx. """
        return NotImplementedError

    @abstractmethod
    def preprocess_image(self, image, label):
        """ Preprocess a single image. """
        return NotImplementedError

    @abstractmethod
    def preprocess_data(self, data, labels):
        """ Preprocess the presented data. """
        preprocessed = [self.preprocess_image(image, label) for image, label in np.column_stack(data, labels)]

        return preprocessed[:, 0], preprocessed[:, 1]


def preprocess_imagenet(image, label):
    image = tf.io.read_file(image)

    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])

    image = tf.keras.applications.vgg16.preprocess_input(image)

    return image, tf.one_hot(label, depth=1000)


class ImagenetDataloader(Dataloader):

    def __init__(self, datapath, partition, batch_size):
        """ Initialize Imagenet Dataloader. """
        super().__init__(datapath, partition, batch_size)

        self.label_dict = pd.read_csv(self.datapath + "imagenet1000_clsid_to_labels.txt", delimiter=":", header=None)

        if not os.path.isdir(self.datapath):
            raise ValueError("The data path does not exist or is not a directory.")

        self.samples = []
        self.labels = []
        for root, dirs, files in os.walk(self.datapath + self.partition):
            for filename in files:
                label_str = root.split("/")[-1]
                label = self.label_dict[self.label_dict[0] == label_str].index[0]

                self.samples.append(root + "/" + filename)
                self.labels.append(label)

    def __len__(self):
        """ Returns the length of the dataset. """
        return len(self.samples)

    def __getitem__(self, index):
        """ Get the datapoint at index. """
        return self.samples[index], self.labels[index]

    def get_dataset_partition(self, startidx, endidx, batched=False):
        """ Returns a dataset partition ranging from startidx to endidx. """
        datapartition = np.array(self.samples[startidx:endidx])
        labelpartition = np.array(self.labels[startidx:endidx])

        if batched:
            # divide data and labels into batches
            datapartition = datapartition.reshape((-1, self.batch_size))
            labelpartition = labelpartition.reshape((-1, self.batch_size))

        return datapartition, labelpartition

    def preprocess_image(self, image, label):
        """ Preprocess a single image. """
        image = tf.io.read_file(image)

        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224, 224])

        image = tf.keras.applications.vgg16.preprocess_input(image).numpy()

        return image, tf.one_hot(label, depth=1000).numpy()

    def preprocess_label(self, label):
        """ Convert label to one hot encoding. """
        return tf.one_hot(label, depth=1000).numpy()

    def preprocess_data(self, data, labels):
        """ Preprocess the given data. """
        prep_images = []
        prep_labels = []

        for image, label in np.column_stack((data, labels)):
            prep_image, prep_label = self.preprocess_image(image, label)
            prep_images.append(prep_image)
            prep_labels.append(prep_label)
        # preprocessed = np.array([self.preprocess_image(image, label) for image, label in np.column_stack((data, labels))])
        # return preprocessed[:, 0], preprocessed[:, 1]
        return np.array(prep_images), np.array(prep_labels)

    def get_data(self, partition, shuffle=False, seed=None):
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

        if shuffle:
            if seed:
                dataset = dataset.shuffle(1000, seed=seed)
            else:
                dataset = dataset.shuffle(1000)

        dataset = dataset.batch(batch_size=self.batch_size)

        return dataset

    def get_data_partition(self, partition, start_index, end_index, shuffle=False, seed=None):
        """ Get a partition of the dataset starting from start_index to end_index"""

        dataset = self.get_data(partition, shuffle=shuffle, seed=seed)

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

    def get_bounding_boxes(self, partition):
        """ Get the bounding boxes."""
        pass


class VocDataloader(Dataloader):

    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor']

    def parse_data(filename, label):
        """ Load and normalize image """
        image_string = tf.io.read_file(filename)
        image_decoded = tf.io.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize(image_decoded, [224, 224])
        image_normalized = image_resized / 127.5 - 1.0

        return image_normalized, label

    def get_data(self, partition, shuffle=False, seed=None):

        """ Load the pascal voc 2012 dataset """
        # get filenames and read label files
        # path = "../../data/VOC2012/"

        #classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        #           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
        #           'tvmonitor']

        filenames = []
        labels = []

        f = open(self.datapath + "ImageSets/Main/" + partition + ".txt", "r")
        for line in f:
            print(line)
            filenames.append(line + ".jpg")

            label = np.zeros(20)
            # parse annotations
            tree = xml.etree.ElementTree.parse(self.datapath + "Annotations/" + line[:-1] + ".xml")
            xml_data = tree.getroot()
            xmlstr = xml.etree.ElementTree.tostring(xml_data, encoding="utf-8", method="xml")
            annotation = dict(xmltodict.parse(xmlstr))['annotation']

            objects = annotation["object"]

            if type(objects) != list:
                label[self.classes.index(objects['name'])] = 1

            for object in annotation['object']:
                if type(object) == collections.OrderedDict:
                    label[self.classes.index(object['name'])] = 1

            labels.append(label)

        labels = np.array(labels)
        print(labels.shape)

        dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        # parse dataset
        dataset = dataset.map(self.parse_data)

        return dataset

    def get_bounding_boxes(self, partition):
        """ read bounding box frames """

        f = open(self.datapath + "ImageSets/Main/" + partition + ".txt", "r")
        for line in f:
            # parse annotations
            tree = xml.etree.ElementTree.parse(self.datapath + "Annotations/" + line[:-1] + ".xml")
            xml_data = tree.getroot()
            xmlstr = xml.etree.ElementTree.tostring(xml_data, encoding="utf-8", method="xml")
            annotation = dict(xmltodict.parse(xmlstr))['annotation']

            objects = annotation["object"]

            if type(objects) != list:
                label[self.classes.index(objects['name'])] = 1
                objects['bndbox']

            for object in annotation['object']:
                if type(object) == collections.OrderedDict:
                    label[self.classes.index(object['name'])] = 1