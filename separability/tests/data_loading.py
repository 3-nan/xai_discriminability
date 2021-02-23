import tensorflow as tf
import numpy as np
import pandas as pd
import xmltodict
import xml
import collections
import os


def load_cifar10():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    for c in range(10):
        indices = (y_train == c).flatten()
        print(len(indices))
        print(indices[:3])
        print(indices.shape)
        print((x_train[indices]).shape)


def scale(ds):
    return ds.map(lambda x, y: (x/255., y))


def normalize(ds):
    return ds.map(lambda x, y: ((x - [0.4076, 0.458, 0.485])/0.5, y))


def load_adience():

    path = "../../data/Adience/reformatted/gender/val"

    dataset = tf.keras.preprocessing.image_dataset_from_directory(path, image_size=(224, 224))

    print(len(dataset))

    dataset = dataset.apply(scale)
    dataset = dataset.apply(normalize)

    # = dataset.map(lambda x, y: ((dataset / (-0.5) - (-0.4076, -0.4580, -0.4850)), y))
    # for batch in dataset:
    #     print(batch[0].shape)
    #     '''
    #     batch[0] = tf.nn.batch_norm_with_global_normalization(
    #         batch[0],
    #         mean=[0.4076, 0.4580, 0.4850],
    #         variance=[0.5, 0.5, 0.5],
    #         beta=[0.0, 0.0, 0.0],
    #         gamma=[1.0, 1.0, 1.0],
    #         variance_epsilon=0.0001, # (0.0, 0.0, 0.0),
    #         scale_after_normalization=False)
    #     '''
    #     #batch = (batch[0]/255., batch[1])
    #     batch = (tf.nn.batch_normalization(batch[0],
    #                                          mean=tf.constant([0.4076, 0.4580, 0.4850]),
    #                                          variance=tf.constant([0.5, 0.5, 0.5]),
    #                                          offset=0.0,
    #                                          scale=0.0,
    #                                          variance_epsilon=0.00001),
    #              batch[1])

    #for batch in dataset:
    #    print(np.max(batch[0]))

    return dataset


#dataset = load_adience()

#model = tf.keras.models.load_model("../models/vgg16_adience/model")

#model.compile(optimizer=tf.keras.optimizers.Adam(), metrics="accuracy")

#score = model.evaluate(dataset)
#print(score)

def parse_function(filename, label):
    """ Load and normalize image """
    image_string = tf.io.read_file(filename)
    image_decoded = tf.io.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize(image_decoded, [224, 224])
    image_normalized = image_resized / 127.5 - 1.0

    return image_normalized, label


def load_voc2012():
    """ Load the pascal voc 2012 dataset """
    # get filenames and read label files
    path = "../../data/VOC2012/"
    label_encoding = dict({0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle',
                           5: 'bus', 6: 'car', 7: 'cat', 8: 'chair', 9: 'cow',
                           10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike', 14: 'person',
                           15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'})
    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    filenames = []
    labels = []

    f = open(path + "ImageSets/Main/train.txt", "r")
    for line in f:
        print(line)
        filenames.append(line + ".jpg")

        label = np.zeros(20)
        # parse annotations
        tree = xml.etree.ElementTree.parse(path + "Annotations/" + line[:-1] + ".xml")
        xml_data = tree.getroot()
        xmlstr = xml.etree.ElementTree.tostring(xml_data, encoding="utf-8", method="xml")
        annotation = dict(xmltodict.parse(xmlstr))['annotation']

        objects = annotation["object"]

        if type(objects) != list:
            label[classes.index(objects['name'])] = 1

        for object in annotation['object']:
            if type(object) == collections.OrderedDict:
                print(object)
                print(object['name'])
                print(classes.index(object['name']))
                label[classes.index(object['name'])] = 1

        print(label)

        labels.append(label)

    labels = np.array(labels)
    print(labels.shape)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    # parse dataset
    dataset = dataset.map(parse_function)

    return dataset


# dataset = load_voc2012()


def load_imagenet_manual():

    path = "../../data/imagenet/train"

    # label_dict = pd.read_csv("../../data/map_clsloc.txt", delimiter=" ", header=None, names=["index", "name", "description"], index_col=0)
    label_dict = pd.read_csv("../../data/imagenet1000_clsid_to_labels.txt", delimiter=":", header=None)

    image_paths  = []
    labels = []

    for root, dirs, files in os.walk(path):
        print(root)
        for filename in files:
            # print(filename)
            label_str = root.split("/")[-1]

            # label = label_dict.loc[label_str]['name']
            label = label_dict[label_dict[0] == label_str].index[0]

            image_paths.append(root + "/" + filename)
            labels.append(label)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    return dataset


def preprocess_imagenet(image, label):

    image = tf.io.read_file(image)

    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])

    image = tf.keras.applications.vgg16.preprocess_input(image)
    # image = tf.keras.applications.resnet.preprocess_input(image)

    return image, tf.one_hot(label, depth=1000)


# dataset = load_imagenet_manual()
# dataset = dataset.map(preprocess_imagenet)
# dataset = list(dataset.as_numpy_iterator())
# print(dataset[0][0].shape)
# print(dataset[:,0].shape)
# dataset = dataset.batch(32)
#
#dataset = dataset.take(1)

# print(dataset.element_spec)
# for element in dataset.take(1):
#     print(element)
#
# resnet = tf.keras.applications.resnet.ResNet50(include_top=True, weights='imagenet')
# vgg16 = tf.keras.applications.VGG16(include_top=True, weights=None, classes=1000, classifier_activation='softmax')
#
# vgg16.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy", "top_k_categorical_accuracy"])
# score = vgg16.evaluate(dataset)
# print(score)
#
# vgg16.save("../models/vgg16_untrained/")

# model = tf.keras.models.load_model("../models/vgg16_imagenet")
# for layer in model.layers:
#     print(layer.name)
#     print(hasattr(layer, 'kernel_initializer'))
#
# print([layer.name for layer in model.layers if hasattr(layer, 'kernel_initializer')])
