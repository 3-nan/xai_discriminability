import os
import sys
import collections
import numpy as np
import pandas as pd
import tensorflow as tf
from xml.etree import ElementTree
import xmltodict

from .datasetinterface import DataSample, Dataset


def get_dataset(classname):
    return getattr(sys.modules[__name__], classname)


class ImagenetSample(DataSample):
    """ Implements an imagenet datasample"""

    def __init__(self, image, filename, label, one_hot_label):
        """ Initialize a single imagenet datasample. """
        super().__init__(image, filename)
        # self.image = image
        # self.filename = filename
        self.label = label
        self.one_hot_label = one_hot_label


class ImagenetDataset(Dataset):

    def __init__(self, datapath, partition):
        """ Initialize Imagenet Dataloader. """
        super().__init__(datapath, partition)

        self.label_dict = pd.read_csv(self.datapath + "imagenet1000_clsid_to_labels.txt", delimiter=":", header=None)

        if not os.path.isdir(self.datapath):
            raise ValueError("The data path does not exist or is not a directory.")

        self.labels = []
        for root, dirs, files in os.walk(self.datapath + self.partition):
            for filename in files:
                label_str = root.split("/")[-1]
                label = self.label_dict[self.label_dict[0] == label_str].index[0]

                self.samples.append(root + "/" + filename)
                self.labels.append(label)

    def __getitem__(self, index):
        """ Get the datapoint at index. """

        filename = self.samples[index]
        label = self.labels[index]

        if self.mode in ["preprocessed"]:
            image = self.preprocess_image(filename)
            one_hot_label = self.preprocess_label(label)
        else:
            image = None
            one_hot_label = None

        sample = ImagenetSample(
            image,
            filename,
            label,
            one_hot_label
        )

        return sample

    def preprocess_image(self, image):
        """ Preprocess a single image. """
        image = tf.io.read_file(image)

        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224, 224])

        image = tf.keras.applications.vgg16.preprocess_input(image).numpy()

        return image

    def preprocess_label(self, label):
        """ Convert label to one hot encoding. """
        return tf.one_hot(label, depth=1000).numpy()

    def get_bounding_boxes(self, partition):
        """ Get the bounding boxes."""
        pass


class VOC2012Sample(DataSample):
    """ Implements a pascal voc 2012 sample. """

    def __init__(self, image, filename, label, one_hot_label):
        super().__init__(image, filename)
        self.label = label
        self.one_hot_label = one_hot_label


class VOC2012Dataset(Dataset):
    """ Implements the pascal voc 2012 dataset. """

    def __init__(self, datapath, partition):
        """ Initialize pascal voc 2012 dataset. """
        super().__init__(datapath, partition)

        self.cmap = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                     'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                     'tvmonitor']
        self.labels = []

        f = open(datapath + "ImageSets/Main/" + partition + ".txt", "r")

        for line in f:
            # get image filepath
            self.samples.append(datapath + "JPEGImages/" + line[:-1] + ".jpg")

            # parse annotations
            tree = ElementTree.parse(datapath + "Annotations/" + line[:-1] + ".xml")
            xml_data = tree.getroot()
            xmlstr = ElementTree.tostring(xml_data, encoding="utf-8", method="xml")
            annotation = dict(xmltodict.parse(xmlstr))['annotation']

            objects = annotation["object"]

            if type(objects) != list:
                self.labels.append([objects['name']])

            else:
                label = []
                for object in annotation['object']:
                    if type(object) == collections.OrderedDict:
                        if object['name'] not in label:
                            label.append(object['name'])

                self.labels.append(label)

    def __getitem__(self, index):
        """ Get the datapoint at index. """

        filename = self.samples[index]
        label = self.labels[index]

        if self.mode in ["preprocessed"]:
            image = self.preprocess_image(filename)
            one_hot_label = self.preprocess_label(label)
        else:
            image = None
            one_hot_label = None

        sample = ImagenetSample(
            image,
            filename,
            label,
            one_hot_label
        )

        return sample

    def preprocess_image(self, image):

        image_string = tf.io.read_file(image)
        image_decoded = tf.io.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize(image_decoded, [224, 224]).numpy()
        image_normalized = image_resized / 127.5 - 1.0

        return image_normalized

    def preprocess_label(self, label):
        """ Convert label to one hot encoding. """
        one_hot_label = np.zeros(len(self.cmap))

        for classname in label:
            one_hot_label[self.cmap.index(classname)] = 1

        return one_hot_label

    def get_bounding_boxes(self, partition):
        """ Get the bounding boxes."""
        pass
