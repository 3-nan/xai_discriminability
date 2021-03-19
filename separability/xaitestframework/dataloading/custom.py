import os
import sys
import collections
import numpy as np
import pandas as pd
import cv2
import json
# import tensorflow as tf
from xml.etree import ElementTree
import xmltodict

from .datasetinterface import DataSample, Dataset
from ..helpers.universal_helper import extract_filename


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

        self.classes = np.unique(self.labels)

    def __getitem__(self, index):
        """ Get the datapoint at index. """

        filename = self.samples[index]
        label = [self.labels[index]]

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
        if isinstance(label, list):
            label = label[0]
        return tf.one_hot(label, depth=1000).numpy()

    def get_bounding_boxes(self, partition):
        """ Get the bounding boxes."""
        pass


class VOC2012Sample(DataSample):
    """ Implements a pascal voc 2012 sample. """

    def __init__(self, datum, filename, label, one_hot_label, binary_mask):
        super().__init__(datum, filename)
        self.label = label
        self.one_hot_label = one_hot_label
        self.binary_mask = binary_mask


class VOC2012Dataset(Dataset):
    """ Implements the pascal voc 2012 dataset. """

    def __init__(self, datapath, partition, classidx=None):
        """ Initialize pascal voc 2012 dataset. """
        super().__init__(datapath, partition)

        self.cmap = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                     'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                     'tvmonitor']
        if not classidx:
            self.classes = self.cmap

        else:
            self.classes = []
            for idx in classidx:
                self.classes.append(self.cmap[int(idx)])

        self.labels = []

        if not classidx:
            f = open(datapath + "ImageSets/Main/" + partition + ".txt", "r")
        else:
            f = []
            for idx in classidx:
                with open(datapath + "ImageSets/Main/" + self.cmap[int(idx)] + "_" + partition + ".txt", "r") as classfile:
                    for line in classfile:
                        filename, in_class = [value for value in line.split(" ") if value]
                        if in_class.startswith("1") and (filename not in f):
                            f.append(filename)

        for line in f:
            if line.endswith("\n"):
                line = line[:-1]
            # get image filepath
            self.samples.append(datapath + "JPEGImages/" + line + ".jpg")

            # parse annotations
            tree = ElementTree.parse(datapath + "Annotations/" + line + ".xml")
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
            binary_mask = None
        elif self.mode == "binary_mask":
            image = None,
            one_hot_label = None,
            binary_mask = self.preprocess_binary_mask(filename)
        else:
            image = None
            one_hot_label = None
            binary_mask = None

        sample = VOC2012Sample(
            image,
            filename,
            label,
            one_hot_label,
            binary_mask
        )

        return sample

    def classname_to_idx(self, class_name):
        """ convert a classname to an index. """
        return self.cmap.index(class_name)

    def preprocess_image(self, image):

        read_image = cv2.imread(image, cv2.IMREAD_COLOR)
        image_resized = cv2.resize(read_image, (224, 224), interpolation=cv2.INTER_CUBIC)
        image_normalized = image_resized.astype(np.float32) / 127.5 - 1.0

        # image_string = tf.io.read_file(image)
        # image_decoded = tf.io.decode_jpeg(image_string, channels=3)
        # image_resized = tf.image.resize(image_decoded, [224, 224]).numpy()
        # image_normalized = image_resized / 127.5 - 1.0

        return image_normalized

    def preprocess_label(self, label):
        """ Convert label to one hot encoding. """
        one_hot_label = np.zeros(len(self.cmap))

        for classname in label:
            one_hot_label[self.cmap.index(classname)] = 1

        return one_hot_label

    def preprocess_binary_mask(self, filename):
        """ Get the bounding box as binary mask."""

        binary_mask = {}
        filename = extract_filename(filename)

        # parse annotations
        tree = ElementTree.parse(self.datapath + "Annotations/" + filename + ".xml")
        xml_data = tree.getroot()
        xmlstr = ElementTree.tostring(xml_data, encoding="utf-8", method="xml")
        annotation = dict(xmltodict.parse(xmlstr))['annotation']

        width = int(annotation["size"]["width"])
        height = int(annotation["size"]["height"])

        # iterate objects
        objects = annotation["object"]

        if type(objects) != list:
            # self.labels.append([objects['name']])
            mask = np.zeros((width, height), dtype=int)

            mask[int(objects['bndbox']['xmin']):int(objects['bndbox']['xmax']), int(objects['bndbox']['ymin']):int(objects['bndbox']['ymax'])] = 1

            binary_mask[objects['name']] = mask

        else:
            for object in annotation['object']:
                if type(object) == collections.OrderedDict:
                    if object['name'] in binary_mask.keys():
                        mask = binary_mask[object['name']]
                    else:
                        mask = np.zeros((width, height), dtype=np.uint8)

                    mask[int(object['bndbox']['xmin']):int(object['bndbox']['xmax']), int(object['bndbox']['ymin']):int(object['bndbox']['ymax'])] = 1

                    binary_mask[object['name']] = mask

        # preprocess binary masks to fit shape of image data
        for key in binary_mask.keys():
            # binary_mask[key] = tf.image.resize(binary_mask[key][:, :, np.newaxis], [224, 224]).numpy().astype(int)
            binary_mask[key] = cv2.resize(binary_mask[key], (224, 224), interpolation=cv2.INTER_NEAREST).astype(np.int)[:, :, np.newaxis]

        return binary_mask


class COCOSample(DataSample):
    """ Implements a coco 2017 sample. """

    def __init__(self, datum, filename, label, one_hot_label, binary_mask):
        super().__init__(datum, filename)
        self.label = label
        self.one_hot_label = one_hot_label
        self.binary_mask = binary_mask


# class COCODataset(Dataset):
#     """ Implements the coco 2017 dataset. """
#
#     def __init__(self, datapath, partition, classidx=None):
#         """ Initialize coco 2017 dataset. """
#         super().__init__(datapath, partition)
#
#         annotation_file = os.path.join(datapath, "annotations", "instances_{}2017.json".format(partition))
#         annotations = json.loads(annotation_file)
#
#         # self.cmap = annotations["categories"]
#         #
#         # if not classidx:
#         #     self.classes = self.cmap
#         #
#         # else:
#         #     self.classes = []
#         #     for idx in classidx:
#         #         self.classes.append(self.cmap[int(idx)])
#
#         self.anns, self.cats, self.imgs = {}, {}, {}
#         self.imgToAnns, catToImgs = collections.defaultdict(list), collections.defaultdict(list)
#
#         for ann in annotations["annotations"]:
#             self.imgToAnns[ann["image_id"]].append(ann)
#             self.anns[ann["id"]] = ann
#         for img in annotations["images"]:
#             self.imgs[img["id"]] = img
#
#         for cat in annotations["categories"]:
#             self.cats[cat["id"]] = cat
#
#         for ann in annotations["annotations"]:
#             catToImgs[ann["category_id"]].append(ann["image_id"])
#
#         # for d, datapoint in annotations["images"]:
#         #     self.samples.append(os.path.join(datapath, "{}2017".format(partition), datapoint["file_name"]))
#
#     def __getitem__(self, index):
#         """ Get the datapoint at index. """
#
#         img = self.samples[index]
#         anns = self.imgToAnns[img["id"]]
#
#         if self.mode in ["preprocessed"]:
#             image = self.preprocess_image(img["file_name"])
#             # label, one_hot_label = self.preprocess_label(anns)
#             binary_mask = None
#         elif self.mode == "binary_mask":
#             image = None,
#             # one_hot_label = None,
#             binary_mask = self.preprocess_binary_mask(filename)
#         else:
#             image = None
#             # one_hot_label = None
#             binary_mask = None
#
#         label, one_hot_label = self.preprocess_label(anns)
#
#         sample = VOC2012Sample(
#             image,
#             img["file_name"],
#             label,
#             one_hot_label,
#             binary_mask
#         )
#
#     def preprocess_label(self, anns):
#         """ Reads the annotations and creates label-list and one-hot-encoded label """
#         label = []
#         one_hot_label = np.zeros(len(self.cmap))
#
#         for ann in anns:
#             l = self.cats[ann["category_id"]]
#             if l not in label:
#                 label.append(l)
#             # update one hot
#             one_hot_label[0] = 1             # TODO: category id to index in one hot label
#
#         return label, one_hot_label
