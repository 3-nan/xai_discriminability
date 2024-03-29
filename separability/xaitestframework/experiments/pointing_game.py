import os
import argparse
import numpy as np
import time
import tracemalloc
import pandas as pd
from scipy.ndimage import gaussian_filter

from ..dataloading.custom import get_dataset
from ..dataloading.dataloader import DataLoader
from ..helpers.universal_helper import compute_relevance_path, extract_filename


def get_explanation(relevance_path, data_name, model_name, layer, xai_method, filename, label):
    """ Load explanation for given filename and label. """
    filename = extract_filename(filename)
    explanation_dir = compute_relevance_path(relevance_path, data_name, model_name, layer, xai_method)
    fname = os.path.join(explanation_dir, str(label), filename)

    explanation = np.load(fname + ".npy")

    if len(explanation.shape) == 3:
        explanation = np.mean(explanation, axis=2)   # np.max(explanation, axis=2)
    return explanation


def estimate_pointing_game_score(data_path, data_name, dataset_name, relevance_path, partition, batch_size, model_name,
                                 layer_names, xai_method, output_dir, gaussian_blur=True):
    """ Computes the pointing game score score. """

    print("Gaussian Blur: {}".format(gaussian_blur))
    
    if isinstance(layer_names, list):
        input_layer = layer_names[0]
    else:
        input_layer = layer_names

    scores = {}
    scores_u50 = {}
    scores_u25 = {}

    # initialize dataset
    print("Loading data from {}, with partition {}".format(os.path.join(data_path, data_name), partition))
    datasetclass = get_dataset(dataset_name)
    dataset = datasetclass(os.path.join(data_path, data_name), partition)
    dataset.set_mode("raw")

    # iterate classes
    for label in dataset.classes:

        print("Start estimation for label {}".format(label))

        classidx = str(dataset.classname_to_idx(label))

        # initialize class score
        class_score = 0
        class_score_u50 = []
        class_score_u25 = []

        # initialize class dataset
        class_data = datasetclass(os.path.join(data_path, data_name), "val", classidx=[classidx])
        class_data.set_mode("binary_mask")
        print("Dataset loaded.")

        dataloader = DataLoader(class_data, batch_size=batch_size)

        for batch in dataloader:

            for sample in batch:
                # get attribution to classidx
                explanation = get_explanation(relevance_path, data_name, model_name, input_layer, xai_method,
                                              sample.filename, classidx)

                if gaussian_blur:
                    # add Gaussian Blur as advised in Zhang et al., 2018
                    explanation = gaussian_filter(explanation, sigma=0.02 * np.max(explanation))

                # find index of max value
                maxindex = np.where(explanation == np.max(explanation))

                binary_mask = sample.binary_mask[label]

                ratio = np.sum(binary_mask) / float(binary_mask.shape[0] * binary_mask.shape[1])

                # check if maximum of explanation is on target object class
                # case max is at more than one pixel
                if len(maxindex[0]) > 1:
                    is_in = 0
                    for pixel in maxindex:
                        is_in = is_in or binary_mask[pixel[0], pixel[1]]
                # print(binary_mask[maxindex[0], maxindex[1]])
                else:
                    is_in = binary_mask[maxindex[0], maxindex[1]]

                if type(is_in) != int:
                    is_in = int(is_in)

                class_score += is_in
                if ratio < 0.5:
                    class_score_u50.append(is_in)
                    if ratio < 0.25:
                        class_score_u25.append(is_in)

        print("Class score: {}".format(class_score))
        print("Class length: {}".format(len(class_data)))
        scores[classidx] = float(class_score) / len(class_data)
        scores_u50[classidx] = np.mean(class_score_u50) if class_score_u50 else None
        scores_u25[classidx] = np.mean(class_score_u25) if class_score_u25 else None

    # save results
    results = []
    for key in scores:
        results.append([data_name, model_name, xai_method,
                        str(key), str(scores[key]), str(scores_u50[key]), str(scores_u25[key])])

    df = pd.DataFrame(results,
                      columns=['dataset', 'model', 'method', 'classidx', 'score', 'score_u50', 'score_u25'])
    if gaussian_blur:
        df.to_csv(os.path.join(output_dir, "{}_{}_{}_blur.csv".format(data_name, model_name, xai_method)), index=False)
    else:
        df.to_csv(os.path.join(output_dir, "{}_{}_{}.csv".format(data_name, model_name, xai_method)), index=False)

    return None


if __name__ == "__main__":

    def bool_is_true(input):
        return input == "True"

    # Setting up an argument parser for command line calls
    parser = argparse.ArgumentParser(description="Test and evaluate multiple xai methods")

    parser.add_argument("-d", "--data_path", type=str, default=None, help="data path")
    parser.add_argument("-dn", "--data_name", type=str, default=None, help="The name of the dataset to be used")
    parser.add_argument("-dl", "--dataset_name", type=str, default=None,
                        help="The name of the dataloader class to be used.")
    parser.add_argument("-rd", "--relevance_datapath", type=str, default=None, help="data folder of relevance maps")
    parser.add_argument("-o", "--output_dir", type=str, default="./output",
                        help="Sets the output directory for the results")
    parser.add_argument("-m", "--model_path", type=str, default=None, help="path to the model")
    parser.add_argument("-mn", "--model_name", type=str, default=None, help="Name of the model to be used")
    parser.add_argument("-mt", "--model_type", type=str, default=None, help="AI Framework to use (tensorflow, pytorch")
    parser.add_argument("-si", "--start_index", type=int, default=0, help="Index of dataset to start with")
    parser.add_argument("-ei", "--end_index", type=int, default=50000, help="Index of dataset to end with")
    parser.add_argument("-p", "--partition", type=str, default="train",
                        help="Either train or val for one of these partitions")
    parser.add_argument("-cl", "--class_label", type=int, default=0, help="Index of class to compute heatmaps for")
    parser.add_argument("-r", "--rule", type=str, default="LRPSequentialCompositeA",
                        help="Rule to be used to compute relevance maps")
    parser.add_argument("-l", "--layer", type=str, default=None, help="Layer to compute relevance maps for")
    parser.add_argument("-bs", "--batch_size", type=int, default=50, help="Batch size for relevance map computation")
    parser.add_argument("-gb", "--gaussian_blur", type=bool_is_true, default=True, help="Boolean indicating if gaussian blur should be applied to attribution")
    parser.add_argument("-x", "--directory", type=str, default=None, help="local_directory")

    ARGS = parser.parse_args()

    #####################
    #       MAIN
    #####################

    print("start pointing game score estimation now")
    start = time.process_time()
    tracemalloc.start()

    estimate_pointing_game_score(ARGS.directory,
                                 ARGS.data_name,
                                 ARGS.dataset_name,
                                 ARGS.directory,
                                 ARGS.partition,
                                 ARGS.batch_size,
                                 ARGS.model_name,
                                 ARGS.layer,
                                 ARGS.rule,
                                 os.path.join(ARGS.directory, "results"),
                                 gaussian_blur=ARGS.gaussian_blur)

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
    tracemalloc.stop()
    print("Duration of pointing game score estimation:")
    print(time.process_time() - start)
    print("Job executed successfully.")
