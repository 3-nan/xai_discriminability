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
parser.add_argument("-o", "--output_dir", type=str, default="/output", help="Sets the output directory for the results")
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


def load_relevance_maps(data_path, start_index, end_index):
    """ Load npy files of relevance maps"""
    rmaps = []
    labels = []

    for dir in os.scandir(data_path + "/train"):
        # collect files and sort
        data_files = os.listdir(dir.path)
        data_files.sort(key=lambda f: int(f.split("_", 1)[0]))

        for file in data_files[:5]:
            load_maps = np.load(dir.path + "/" + file)
            rmaps.append(load_maps)
            labels.append(np.ones(load_maps.shape[0], dtype=np.int)*int(dir.name))

    rmaps = np.concatenate(rmaps)
    labels = np.concatenate(labels)

    # shuffle data
    p = np.random.permutation(len(labels))

    print(rmaps.shape)
    print(labels.shape)

    return rmaps[p], labels[p]


def load_relevance_map_selection(data_path, partition):
    """ Load a selection of relevance maps
    partition: str - either train or test
    """
    rmaps = []
    labels = []

    for dir in os.scandir(data_path + "/" + partition):
        # collect files and sort
        data_files = os.listdir(dir.path)
        data_files.sort(key=lambda f: int(f.split("-", 1)[0]))


# load data
data_path = ARGS.data_path + "/" + ARGS.data_name + "/" + ARGS.model_name + "/" + ARGS.layer + "/" + ARGS.rule

relevance_maps, labels = load_relevance_maps(data_path, 0, 10000)

if len(relevance_maps.shape) == 4:
    relevance_maps = np.reshape(relevance_maps, (relevance_maps.shape[0], relevance_maps.shape[1]*relevance_maps.shape[2]*relevance_maps.shape[3]))

print("start training linear classifier now")
start = time.process_time()

clf = LinearSVC()

clf.fit(relevance_maps, labels)

score = clf.score(relevance_maps, labels)

print("Duration of training of linear clf:")
print(time.process_time() - start)

# output
output_dir = ARGS.output_dir

print("separability score")
print(score)

df = pd.DataFrame([[ARGS.data_name, ARGS.model_name, ARGS.layer, ARGS.rule, str(score)]],
                  columns=['dataset', 'model', 'layer', 'method', 'separability_score'])
df.to_csv(ARGS.output_dir + ARGS.data_name + "_" + ARGS.model_name + "_" + ARGS.layer + "_" + ARGS.rule + ".csv",
          index=False)
