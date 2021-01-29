import argparse
import datetime
import time
import os
import random
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
import tracemalloc

from ..dataloading.custom import get_dataset
from ..dataloading.dataloader import DataLoader
from ..helpers.universal_helper import extract_filename, join_path, compute_relevance_path


def load_explanations(explanationdir, samples, classidx):
    """ Load explanations for the given classidx. """

    explanations = []

    if isinstance(classidx, list):
        for sample in samples:
            explanations.append(np.load(join_path(explanationdir, [str(classidx), sample.filename]) + ".npy"))
    else:
        for s, sample in enumerate(samples):
            explanations.append(np.load(join_path(explanationdir, [str(classidx[s]), sample.filename]) + ".npy"))

    return np.array(explanations)


def load_explanation_data_for_svc(dataloader, classidx, classes, explanationdir):
    """ Load data for the svm classification. """

    data = []
    labels = []

    for batch in dataloader:
        filenames = [sample.filename for sample in batch]

        # target explanations
        explanations = load_explanations(explanationdir, batch, classidx=classidx)

        data.append(explanations)
        labels.append(np.ones(len(explanations)))

        # non-target explanations
        target_classes = random.sample(classes, len(batch))
        explanations = load_explanations(explanationdir, batch, classidx=target_classes)

        data.append(explanations)
        labels.append(np.zeros(len(explanations)))

    data = np.concatenate(data)
    labels = np.concatenate(labels)

    return data, labels


def load_relevance_map_selection(data_path, partition, label, filenames):
    """ Load a selection of relevance maps
    label: int - class label to get data for
    """
    rmaps = []
    labels = []

    for dir in os.scandir(join_path(data_path, partition)):

        R_c = []

        for filename in filenames:
            R_c.append(np.load(join_path(dir.path, filename) + ".npy"))

        # R_c = np.append(R_c)

        rmaps.append(R_c)

        # add labels for correct / incorrect class to labels
        if dir.name == str(label):
            # append true to labels
            labels.append(np.ones(len(filenames)))
        else:
            # append false to labels
            labels.append(np.zeros(len(filenames)))

    rmaps = np.concatenate(rmaps)
    labels = np.concatenate(labels)

    print(rmaps.shape)
    print(labels.shape)

    # shuffle files
    p = np.random.permutation(len(labels))

    return rmaps[p], labels[p]


def estimate_separability_score(data_path, data_name, dataset_name, relevance_path, partition, batch_size, model_name, layer_name, rule, output_dir):
    """ Compute the separability score for the provided relevances. """

    relevance_path = compute_relevance_path(relevance_path, data_name, model_name, layer_name, rule)

    # initialize dataset
    datasetclass = get_dataset(dataset_name)
    dataset = datasetclass(data_path, "train")
    dataset.set_mode("raw")

    class_indices = [str(dataset.classname_to_idx(name)) for name in dataset.classes]

    # iterate classes
    for classidx in class_indices:

        print("estimate one class separability for class " + classidx)

        # load dataset for this class
        class_data = datasetclass(data_path, partition, classidx=classidx)
        class_data.set_mode("raw")

        dataloader = DataLoader(class_data, batch_size=batch_size, shuffle=True, startidx=0, endidx=1000)

        # load relevance maps from train partition to train clf
        Rc_data, labels = load_explanation_data_for_svc(dataloader, classidx, class_indices,
                                                        join_path(relevance_path, "train"))

        print(Rc_data.shape)
        print(labels.shape)

        if len(Rc_data.shape) == 4:
            Rc_data = np.reshape(Rc_data, (Rc_data.shape[0], Rc_data.shape[1] * Rc_data.shape[2] * Rc_data.shape[3]))

        clf = LinearSVC()
        clf.fit(Rc_data, labels)

        # load test data
        test_filenames = []
        for sample in testset:
            if classlabel in sample.label:
                test_filenames.append(extract_filename(sample.filename))

        Rc_test, test_labels = load_relevance_map_selection(relevance_path, 'val', classlabel, test_filenames)

        if len(Rc_test.shape) == 4:
            Rc_test = np.reshape(Rc_test, (Rc_test.shape[0], Rc_test.shape[1] * Rc_test.shape[2] * Rc_test.shape[3]))

        # compute score
        score = clf.score(Rc_test, test_labels)

        print("separability score for class " + str(classlabel) + " : " + str(dataset.classname_to_idx(classlabel)))
        print(score)

        if not os.path.exists(output_dir + data_name + "_" + model_name):
            os.makedirs(output_dir + data_name + "_" + model_name)

        df = pd.DataFrame([[data_name, model_name, layer_name, rule, str(score)]],
                          columns=['dataset', 'model', 'layer', 'method', 'separability_score'])
        df.to_csv(output_dir + data_name + "_" + model_name + "/" + layer_name + "_" + rule + "_"
                  + str(dataset.classname_to_idx(classlabel)) + ".csv", index=False)


current_datetime = datetime.datetime.now()
print(current_datetime)

print("one_class_separability")

# Setting up an argument parser for command line calls
parser = argparse.ArgumentParser(description="Test and evaluate multiple xai methods")

parser.add_argument("-d", "--data_path", type=str, default=None, help="data path")
parser.add_argument("-dn", "--data_name", type=str, default=None, help="The name of the dataset to be used")
parser.add_argument("-dl", "--dataloader_name", type=str, default=None, help="The name of the dataloader class to be used.")
parser.add_argument("-rd", "--relevance_datapath", type=str, default=None, help="data folder of relevance maps")
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

print("start separability score estimation now")
start = time.process_time()
tracemalloc.start()

estimate_separability_score(ARGS.data_path,
                            ARGS.data_name,
                            ARGS.dataloader_name,
                            ARGS.relevance_datapath,
                            ARGS.partition,
                            ARGS.batch_size,
                            ARGS.model_name,
                            ARGS.layer,
                            ARGS.rule,
                            ARGS.output_dir)

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
tracemalloc.stop()
print("Duration of separability score estimation:")
print(time.process_time() - start)
