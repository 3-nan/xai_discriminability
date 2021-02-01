import argparse
import time
import numpy as np
import pandas as pd
import tracemalloc

from ..dataloading.custom import get_dataset
from ..dataloading.dataloader import DataLoader
from ..helpers.model_helper import init_model
from ..helpers.universal_helper import compute_relevance_path, extract_filename, join_path


def load_explanations(explanationdir, samples, classidx):
    """ Load explanations for the given classidx. """

    explanations = []

    explanationdir = join_path(explanationdir, str(classidx))

    for sample in samples:
        explanations.append(np.load(join_path(explanationdir, extract_filename(sample.filename)) + ".npy"))

    return np.array(explanations)


def compute_pixelflipping_score(data_path, data_name, dataset_name, relevance_path, partition, batch_size, model_path, model_name, layer_name, rule, distribution, output_dir):
    """ Estimate the pixelflipping score. """

    flip_percentages = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                        0.9]

    # construct explanationpath
    explanationdir = compute_relevance_path(relevance_path, data_name, model_name, layer_name, rule)
    explanationdir = join_path(explanationdir, partition)

    # init model
    model = init_model(model_path)

    # load dataset
    datasetclass = get_dataset(dataset_name)
    dataset = datasetclass(data_path, partition)

    for classname in dataset.classes:

        classidx = dataset.classname_to_idx(classname)

        # prep result structure
        class_score = {}
        for percentage in flip_percentages:
            class_score[percentage] = []

        # load dataset for this class
        class_data = datasetclass(data_path, partition, classidx=classidx)
        class_data.set_mode("preprocessed")

        dataloader = DataLoader(class_data, batch_size=batch_size)

        # iterate data
        for batch in dataloader:

            data = [sample.image for sample in batch]

            # get/sort indices for pixelflipping order
            explanations = load_explanations(explanationdir, batch, classidx)
            # reduce explanations dimension
            explanations = np.max(explanations, axis=3)

            print(explanations.shape)

            if rule in ["Gradient", "SmoothGrad", "LRPZ"]:
                indices = [np.argsort(np.abs(explanation), axis=None) for explanation in explanations]
            else:
                indices = [np.argsort(explanation, axis=None) for explanation in explanations]

            indices = np.array(indices)

            print(indices.shape)

            # loop flip_percentages
            for percentage in flip_percentages:

                # get first percentage part of pixel indices (lowest relevance)
                # indicesfraction = indices[:, :int(flip_percentage * len(indices))]
                # get last percentage part of pixel indices (highest relevance)
                indicesfraction = indices[:, int((1 - percentage) * len(indices)):]

                flipped_data = []

                # flip images
                for p, point in enumerate(data):
                    # flip pixels
                    for axis in range(point.shape[2]):
                        if distribution == "uniform":
                            random_values = np.random.uniform(-1.0, 1.0, len(indicesfraction[p]))
                        elif distribution == "gaussian":
                            random_values = np.random.normal(loc=0.0, scale=1.0, size=len(indicesfraction[p]))
                        else:
                            raise ValueError("No distribution for flipping pixels specified.")
                        np.put_along_axis(point[:, :, axis], indicesfraction[p], random_values, axis=None)

                    flipped_data.append(point)

                flipped_data = np.array(flipped_data)

                # compute score on flipped data
                predictions = model.predict(flipped_data, batch_size=len(flipped_data))

                class_score[percentage].append(predictions[:, classidx])

        # collect results and write to file
        results = []
        for key in class_score:
            results.append([data_name, model_name, rule, str(key), str(np.mean(class_score[key]))])

        df = pd.DataFrame(results, columns=['dataset', 'model', 'method', 'flip_percentage', 'flipped_score'])
        df.to_csv(output_dir + data_name + "_" + model_name + "_" + rule + "_" + distribution + str(classidx) + ".csv", index=False)


if __name__ == "__main__":
    # Setting up an argument parser for command line calls
    parser = argparse.ArgumentParser(description="Test and evaluate multiple xai methods")

    parser.add_argument("-d", "--data_path", type=str, default=None, help="data path")
    parser.add_argument("-dn", "--data_name", type=str, default=None, help="The name of the dataset to be used")
    parser.add_argument("-dl", "--dataloader_name", type=str, default=None, help="The name of the dataloader class to be used.")
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
    parser.add_argument("-pd", "--distribution", type=str, default="", help="Probability distribution to sample flipped pixels from (uniform, gaussian)")

    ARGS = parser.parse_args()

    #####################
    #       MAIN
    #####################

    print("start pixelflipping now")
    start = time.process_time()
    tracemalloc.start()

    compute_pixelflipping_score(ARGS.data_path,
                                ARGS.data_name,
                                ARGS.dataloader_name,
                                ARGS.relevance_datapath,
                                ARGS.partition,
                                ARGS.batch_size,
                                ARGS.model_path,
                                ARGS.model_name,
                                ARGS.layer,
                                ARGS.rule,
                                ARGS.distribution,
                                ARGS.output_dir)

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    tracemalloc.stop()
    print("Duration of pixelflipping estimation:")
    print(time.process_time() - start)
