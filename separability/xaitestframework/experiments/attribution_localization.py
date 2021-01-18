import argparse
import numpy as np
import time
import tracemalloc
import pandas as pd

from ..dataloading.custom import get_dataset
from ..dataloading.dataloader import DataLoader
from helpers.universal_helper import compute_relevance_path, join_path, extract_filename


def get_explanation(relevance_path, data_name, model_name, layer, xai_method, filename, label):
    """ Load explanation for given filename and label. """
    filename = extract_filename(filename)
    explanation_dir = compute_relevance_path(relevance_path, data_name, model_name, layer, xai_method)
    fname = join_path(explanation_dir, [label, filename])

    explanation = np.load(fname + ".npy")
    return explanation


def attribution_localization(data_path, data_name, dataset_name, relevance_path, partition, batch_size, model_path, model_name, layer_names, xai_method, output_dir):
    """ Computes the attribution localization score. """

    # initialize dataset and dataloader
    dataset = get_dataset(dataset_name)
    dataset = dataset(data_path, "val")
    dataset.set_mode("binary_mask")

    dataloader = DataLoader(dataset, batch_size=batch_size)

    total_scores = []
    weighted_scores = []

    for batch in dataloader:
        for sample in batch:
            sample_score = 0.0
            sample_weighted_score = 0.0
            for label in sample.labels:
                # get attribution according to label
                explanation = get_explanation(relevance_path, data_name, model_name, "input_1", xai_method, sample.filename, label)
                binary_mask = sample.binary_mask[label]

                # compute inside - total relevance ratios
                inside_explanation = np.sum(explanation[binary_mask])
                total_explanation = np.sum(explanation)

                size_bbox = np.sum(binary_mask)
                size_data = np.shape(binary_mask)[0] * np.shape(binary_mask)[1]

                sample_score += inside_explanation / total_explanation
                sample_weighted_score += (inside_explanation / total_explanation) * (size_data / size_bbox)

            total_scores.append(sample_score / len(sample.labels))
            weighted_scores.append(sample_weighted_score / len(sample.labels))

    total_score = float(np.sum(total_scores)) / len(total_scores)
    weighted_score = float(np.sum(weighted_scores) / len(weighted_scores))

    # save results
    results = [data_name, model_name, xai_method, str(total_score), str(weighted_score)]

    df = pd.DataFrame(results,
                      columns=['dataset', 'model', 'method', 'total_score', 'weighted score'])
    df.to_csv(ARGS.output_dir + ARGS.data_name + "_" + ARGS.model_name + "_" + ARGS.rule + ".csv", index=False)


# Setting up an argument parser for command line calls
parser = argparse.ArgumentParser(description="Test and evaluate multiple xai methods")

parser.add_argument("-d", "--data_path", type=str, default=None, help="data path")
parser.add_argument("-dn", "--data_name", type=str, default=None, help="The name of the dataset to be used")
parser.add_argument("-dl", "--dataset_name", type=str, default=None, help="The name of the dataloader class to be used.")
parser.add_argument("-rd", "--relevance_datapath", type=str, default=None, help="data folder of relevance maps")
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

print("start explanation localization score estimation now")
start = time.process_time()
tracemalloc.start()

attribution_localization(ARGS.datapath,
                         ARGS.data_name,
                         ARGS.dataset_name,
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
