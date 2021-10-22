import argparse
import datetime
import time
import os
import shutil
import numpy as np
import pandas as pd
# from sklearn.svm import LinearSVC
from sklearn.metrics import average_precision_score, f1_score
import tracemalloc

from ..dataloading.custom import get_dataset
from ..dataloading.dataloader import DataLoader
from ..helpers.model_helper import init_model
from ..helpers.universal_helper import extract_filename


def scale_attribution(attribution):
    """ Scale the attribution with abs sum. """

    # what to do here?
    # max is arbitrary, so sum is the preferred option
    # best: abs(sum()), but sum might be 0
    # therefore: sum(abs()), but not quite the same..
    # maybe: max(abs())
    # attribution = attribution / np.sum(np.abs(attribution))
    attribution = attribution / np.max(np.abs(attribution))

    return attribution


def compute_attributions_for_batch(batch, model, layer_name, xai_method, classidx, classes, explanationdir):
    """ Compute attributions for data batch. """

    # if str(classidx) in classes:
    #     classindices = classes.copy()
    #     classindices.remove(str(classidx))
    # else:
    #     classindices = classes
    classindices = classes

    # extract preprocessed data
    data = [sample.datum for sample in batch]

    for index in classindices:
        # compute relevance
        R = model.compute_relevance(data, [layer_name], int(index), xai_method,
                                    additional_parameter=None)  # TODO: add additional parameter to pipeline

        layer_output_dir = os.path.join(explanationdir, str(index))
        os.makedirs(layer_output_dir, exist_ok=True)
        # layer_output_dir = combine_path(output_dir, [layer_name, xai_method, partition, str(class_name)])
        for r, relevance in enumerate(R[layer_name]):
            fname = extract_filename(batch[r].filename)
            filename = os.path.join(layer_output_dir, fname + ".npy")
            np.save(filename, relevance)


def load_attributions_for_sample(filename, classidx, classes, explanationdir):
    """ Load data for the svm classification. """

    if str(classidx) in classes:
        classindices = classes.copy()
        classindices.remove(str(classidx))
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


def estimate_separability_score(data_path, data_name, dataset_name, classidx, relevance_path, partition, batch_size, model_path, model_name, model_type, layer_name, xai_method, output_dir):
    """ Compute the separability score for the provided relevances. """

    # layer_name = layer_names[0]     # TODO: iterate layers
    # endidx = 400

    relevance_path = os.path.join(relevance_path, "attributions")
    os.makedirs(relevance_path)

    # init model
    model = init_model(model_path, model_name, framework=model_type)

    # initialize dataset
    datasetclass = get_dataset(dataset_name)
    dataset = datasetclass(os.path.join(data_path, data_name), partition)
    dataset.set_mode("raw")

    print(dataset.classes)
    class_indices = [str(dataset.classname_to_idx(name)) for name in dataset.classes]

    print("estimate one class separability for class {}".format(classidx))

    scores = []
    aps = []
    f1s = []
    aps_robust = []
    f1s_robust = []
    skipped = 0
    apr_isnan = 0

    # load dataset for this class
    class_data = datasetclass(os.path.join(data_path, data_name), partition, classidx=[classidx])
    class_data.set_mode("preprocessed")

    print("number of samples for this class is {}".format(len(class_data)))

    dataloader = DataLoader(class_data, batch_size=batch_size, startidx=0, endidx=100)  # , startidx=0, endidx=2000)

    for batch in dataloader:

        os.makedirs(relevance_path, exist_ok=True)

        compute_attributions_for_batch(batch, model, layer_name, xai_method, classidx, class_indices, relevance_path)

        for sample in batch:

            # load explanations for sample
            Rs, labels = load_attributions_for_sample(sample.filename, classidx, class_indices, relevance_path)

            if len(Rs.shape) == 3:
                Rs = np.reshape(Rs, (Rs.shape[0], Rs.shape[1] * Rs.shape[2]))
            elif len(Rs.shape) == 4:
                Rs = np.reshape(Rs, (Rs.shape[0], Rs.shape[1] * Rs.shape[2] * Rs.shape[3]))

            # clf = LinearSVC(class_weight="balanced")
            # clf = LinearSVC(C=1000, class_weight="balanced", max_iter=100)    # very high C (regularization parameter)
            # clf.fit(Rs, labels)
            #
            # # measure with of margin of trained SVM
            # margin = 1. / np.sqrt(np.sum(clf.coef_ ** 2))
            #
            scores.append(0.0)

            # compute sample weights
            # sample_weights = (1. - np.mean(test_labels)) * test_labels
            # sample_weights[sample_weights == 0.0] = np.mean(test_labels)
            #
            # # compute score
            # score = clf.score(Rc_test, test_labels, sample_weight=sample_weights)

            if (Rs[0] > 0.).any():
                target = np.array(Rs[0] >= 0.)

                # robust_target = np.array(Rs[0] >= 0.1)

                mask = (np.abs(Rs[0]) >= 0.1)

                assert target.shape == Rs[0].shape

                # compute scores
                ap = np.mean([average_precision_score(target, r, average="samples") for r in Rs[1:]])
                f1 = np.mean([f1_score(target, np.array(r >= 0.), average="binary") for r in Rs[1:]])

                # ap_robust = np.mean([average_precision_score(robust_target, r, average="samples") for r in Rs[1:]])
                ap_robust = np.mean([average_precision_score(target[mask], r[mask], average="samples") for r in Rs[1:]])
                # f1_robust = np.mean([f1_score(robust_target, np.array(r >= 0), average="binary") for r in Rs[1:]])
                f1_robust = np.mean([f1_score(target[mask], np.array(r[mask] >= 0.), average="binary") for r in Rs[1:]])

                aps.append(ap)
                f1s.append(f1)

                if np.isnan(ap_robust):
                    apr_isnan += 1
                else:
                    aps_robust.append(ap_robust)
                f1s_robust.append(f1_robust)

            else:
                skipped += 1

        print("Remove computed attributions")
        shutil.rmtree(relevance_path)

    print("separability score for class {}".format(classidx))
    print(np.mean(scores))

    resultdir = os.path.join(output_dir, "{}_{}".format(data_name, model_name))

    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

    df = pd.DataFrame([[data_name, model_name, layer_name, xai_method, str(np.mean(scores)),
                        str(np.mean(aps)), str(np.mean(f1s)), str(np.mean(aps_robust)), str(np.mean(f1s_robust)),
                        str(skipped), str(apr_isnan)]],
                      columns=['dataset', 'model', 'layer', 'method', 'separability_score',
                               'ap', 'f1', 'apr', 'f1r', 'skipped', 'apr_isnan'])
    df.to_csv("{}/{}_{}_{}.csv".format(resultdir, layer_name, xai_method, str(classidx)), index=False)


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
    parser.add_argument("-x", "--directory", type=str, default=None, help="local_directory")

    ARGS = parser.parse_args()

    #####################
    #       MAIN
    #####################

    print("start separability score estimation now")
    start = time.process_time()
    tracemalloc.start()

    estimate_separability_score(ARGS.directory,
                                ARGS.data_name,
                                ARGS.dataloader_name,
                                ARGS.class_label,
                                ARGS.directory,
                                ARGS.partition,
                                ARGS.batch_size,
                                ARGS.model_path,
                                ARGS.model_name,
                                ARGS.model_type,
                                ARGS.layer,
                                ARGS.rule,
                                os.path.join(ARGS.directory, "results"))

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    tracemalloc.stop()
    print("Duration of separability score estimation:")
    print(time.process_time() - start)
    print("Job executed successfully.")
