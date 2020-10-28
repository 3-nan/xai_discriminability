import argparse
import datetime
import time
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.svm import LinearSVC

import innvestigate

current_datetime = datetime.datetime.now()
print(current_datetime)

# Setting up an argument parser for command line calls
parser = argparse.ArgumentParser(description="Test and evaluate multiple xai methods")

parser.add_argument("-d", "--data_path", type=str, default=None, help="data path")
parser.add_argument("-dn", "--data_name", type=str, default=None, help="The name of the dataset to be used")
parser.add_argument("-o", "--output_dir", type=str, default="./output", help="Sets the output directory for the results")
parser.add_argument("-m", "--model_path", type=str, default=None, help="path to the model")
parser.add_argument("-mn", "--model_name", type=str, default=None, help="Name of the model to be used")
parser.add_argument("-si", "--start_index", type=int, default=0, help="Index of dataset to start with")
parser.add_argument("-ei", "--end_index", type=int, default=50000, help="Index of dataset to end with")
parser.add_argument("-p", "--partition", type=str, default="train", help="Either train or test for one of these partitions")
parser.add_argument("-cl", "--class_label", type=int, default=0, help="Index of class to compute heatmaps for")
parser.add_argument("-r", "--rule", type=str, default="LRPSequentialCompositeA", help="Rule to be used to compute relevance maps")
parser.add_argument("-l", "--layer", type=str, default=None, help="Layer to compute relevance maps for")
parser.add_argument("-bs", "--batch_size", type=int, default=50, help="Batch size for relevance map computation")

ARGS = parser.parse_args()

#####################
#       MAIN
#####################


def load_relevance_map_selection(data_path, label, indices):
    """ Load a selection of relevance maps
    label: int - class label to get data for
    """
    rmaps = []
    labels = []

    for dir in os.scandir(data_path + "/" + 'train'):

        # load data from indices
        data_files = os.listdir(dir.path)
        data_files.sort(key=lambda f: int(f.split("_", 1)[0]))

        R_c = []
        for file in data_files:
            R_c.append(np.load(dir.path + "/" + file))

        R_c = np.concatenate(R_c)

        rmaps.append(R_c[indices])

        # get number of entries per data file
        # size = int(data_files[0].split("_", 1)[1].split(".")[0]) * 5

        if dir.name == str(label):
            # append true to labels
            labels.append(np.ones(np.sum(indices)))
        else:
            # append false to labels
            labels.append(np.zeros(np.sum(indices)))

    rmaps = np.concatenate(rmaps)
    labels = np.concatenate(labels)

    print(rmaps.shape)
    print(labels.shape)

    # shuffle files
    p = np.random.permutation(len(labels))

    return rmaps[p], labels[p]


# load data
data_path = ARGS.data_path + "/" + ARGS.data_name + "/" + ARGS.model_name + "/" + ARGS.layer + "/" + ARGS.rule

(_, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
# get selection of data and relevance maps
for c in range(10):

    print("estimate one class separability for class " + str(c))

    indices = (y_train == c).flatten()
    Rc_data, labels = load_relevance_map_selection(data_path, c, indices)

    if len(Rc_data.shape) == 4:
        Rc_data = np.reshape(Rc_data, (Rc_data.shape[0], Rc_data.shape[1] * Rc_data.shape[2] * Rc_data.shape[3]))

    clf = LinearSVC()
    clf.fit(Rc_data[:40000], labels[:40000])

    score = clf.score(Rc_data[40000:], labels[40000:])

    print("separability score for class " + str(c))
    print(score)

    df = pd.DataFrame([[ARGS.data_name, ARGS.model_name, ARGS.layer, ARGS.rule, str(score)]],
                      columns=['dataset', 'model', 'layer', 'method', 'separability_score'])
    df.to_csv(ARGS.output_dir + ARGS.data_name + "_" + ARGS.model_name + "_" + ARGS.layer + "_" + ARGS.rule + "_"
              + str(c) + ".csv", index=False)
