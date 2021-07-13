import argparse
import datetime
import time
import os
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import average_precision_score
import tracemalloc

from ..dataloading.custom import get_dataset
from ..dataloading.dataloader import DataLoader
from ..helpers.universal_helper import extract_filename, compute_relevance_path


def scale_attribution(attribution):
    """ Scale the attribution with abs sum. """

    # what to do here?
    # max is arbitrary, so sum is the preferred option
    # best: abs(sum()), but sum might be 0
    # therefore: sum(abs()), but not quite the same..
    # maybe: max(abs())
    attribution = attribution / np.sum(np.abs(attribution))

    return attribution


def load_attributions_for_sample(filename, classidx, classes, explanationdir):
    """ Load data for the svm classification. """

    if classidx in classes:
        classindices = classes.copy()
        classindices.remove(classidx)
    else:
        classindices = classes

    # if len(explanations.shape) > 3:
    #     explanations = np.mean(explanations, axis=(1, 2))    # axis = 3

    Rs = []
    labels = []

    # target explanation
    attribution = np.load(os.path.join(explanationdir, str(classidx), extract_filename(filename)) + ".npy")

    attribution = scale_attribution(attribution)

    Rs.append(attribution)
    labels.append(1)

    for idx in classindices:
        # non-target explanations
        attribution = np.load(os.path.join(explanationdir, str(idx), extract_filename(filename)) + ".npy")

        attribution = scale_attribution(attribution)

        Rs.append(attribution)
        labels.append(0)

    Rs = np.array(Rs)
    labels = np.array(labels)

    # p = np.random.permutation(len(labels))

    return Rs, labels


def estimate_separability_score(data_path, data_name, dataset_name, relevance_path, partition, batch_size, model_name, layer_name, rule, output_dir):
    """ Compute the separability score for the provided relevances. """

    # layer_name = layer_names[0]     # TODO: iterate layers
    # endidx = 400

    relevance_path = compute_relevance_path(relevance_path, data_name, model_name, layer_name, rule)

    # initialize dataset
    datasetclass = get_dataset(dataset_name)
    dataset = datasetclass(data_path, partition)
    dataset.set_mode("raw")

    print(dataset.classes)
    class_indices = [str(dataset.classname_to_idx(name)) for name in dataset.classes]

    print("Estimate scores for classindices {}".format(class_indices))

    # iterate classes
    for classidx in class_indices:

        print("estimate one class separability for class " + classidx)

        scores = []
        aps = []

        # load dataset for this class
        class_data = datasetclass(data_path, partition, classidx=[classidx])
        class_data.set_mode("raw")

        print("number of samples for this class is {}".format(len(class_data)))

        dataloader = DataLoader(class_data, batch_size=batch_size)  # , startidx=0, endidx=2000)

        for batch in dataloader:

            for sample in batch:

                # load explanations for sample
                Rs, labels = load_attributions_for_sample(sample.filename, classidx, class_indices, os.path.join(relevance_path, partition))

                if len(Rs.shape) == 3:
                    Rs = np.reshape(Rs, (Rs.shape[0], Rs.shape[1] * Rs.shape[2]))
                elif len(Rs.shape) == 4:
                    Rs = np.reshape(Rs, (Rs.shape[0], Rs.shape[1] * Rs.shape[2] * Rs.shape[3]))

                # clf = LinearSVC(class_weight="balanced")
                clf = LinearSVC(C=1000, class_weight="balanced", max_iter=100)    # very high C (regularization parameter)
                clf.fit(Rs, labels)

                # measure with of margin of trained SVM
                margin = 1. / np.sqrt(np.sum(clf.coef_ ** 2))

                scores.append(margin)

                # compute sample weights
                # sample_weights = (1. - np.mean(test_labels)) * test_labels
                # sample_weights[sample_weights == 0.0] = np.mean(test_labels)
                #
                # # compute score
                # score = clf.score(Rc_test, test_labels, sample_weight=sample_weights)

                target = np.array(Rs[0] >= 0.)

                assert target.shape == Rs[0].shape

                ap = np.mean([average_precision_score(target, r, average="samples") for r in Rs[1:]])

                aps.append(ap)



        print("separability score for class {}".format(classidx))
        print(np.mean(scores))

        resultdir = os.path.join(output_dir, "{}_{}".format(data_name, model_name))

        if not os.path.exists(resultdir):
            os.makedirs(resultdir)

        df = pd.DataFrame([[data_name, model_name, layer_name, rule, str(np.mean(scores)), str(np.mean(aps))]],
                          columns=['dataset', 'model', 'layer', 'method', 'separability_score', 'sample_ap'])
        df.to_csv("{}/{}_{}_{}.csv".format(resultdir, layer_name, rule, str(classidx)), index=False)


if __name__ == "__main__":
    current_datetime = datetime.datetime.now()
    print(current_datetime)

    print("one_class_separability")

    # def decode_layernames(string):
    #     """ Decodes the layer_names string to a list of strings. """
    #     return string.split(":")

    # Setting up an argument parser for command line calls
    parser = argparse.ArgumentParser(description="Test and evaluate multiple xai methods")

    parser.add_argument("-d", "--data_path", type=str, default=None, help="data path")
    parser.add_argument("-dn", "--data_name", type=str, default=None, help="The name of the dataset to be used")
    parser.add_argument("-dl", "--dataloader_name", type=str, default=None, help="The name of the dataloader class to be used.")
    parser.add_argument("-rd", "--relevance_datapath", type=str, default=None, help="data folder of relevance maps")
    parser.add_argument("-o", "--output_dir", type=str, default="/output", help="Sets the output directory for the results")
    parser.add_argument("-m", "--model_path", type=str, default=None, help="path to the model")
    parser.add_argument("-mn", "--model_name", type=str, default=None, help="Name of the model to be used")
    parser.add_argument("-mt", "--model_type", type=str, default=None, help="AI Framework to use (tensorflow, pytorch")
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
