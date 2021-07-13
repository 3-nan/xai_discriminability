import os
import argparse
import time
import numpy as np
import pandas as pd
import tracemalloc
import cv2
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski

from ..dataloading.custom import get_dataset
from ..dataloading.dataloader import DataLoader
from ..helpers.model_helper import init_model
from ..helpers.universal_helper import compute_relevance_path, extract_filename


#############################
# pixel flipping variations #
#############################

def simple_sampling(batch, indicesfraction, flipping_method):
    flipped_data = []

    # flip images
    for s, sample in enumerate(batch):
        # flip pixels

        datum = sample.datum
        for axis in range(datum.shape[2]):
            if flipping_method == "uniform":
                random_values = np.random.uniform(-1.0, 1.0, len(indicesfraction[s]))
            elif flipping_method == "gaussian":
                random_values = np.random.normal(loc=0.0, scale=1.0, size=len(indicesfraction[s]))
            else:
                raise ValueError("No distribution for flipping pixels specified.")
            np.put_along_axis(datum[:, :, axis], indicesfraction[s], random_values, axis=None)

        flipped_data.append(datum)

    flipped_data = np.array(flipped_data)
    return flipped_data


def region_perturbation(batch, indicesfraction, flipping_method):
    indices = []

    # flip images
    for s, sample in enumerate(batch):

        sample_indices = []

        shape = sample.datum.shape

        # add 9x9 region for each pixel
        for pixel in indicesfraction[s]:

            mod = pixel % shape[0]

            # add left pixels
            if mod != 0:
                # print(type(sample_indices))
                # print(sample_indices)
                # print(type(pixel))
                # print(pixel)
                sample_indices += [pixel - shape[0] - 1, pixel - 1, pixel + shape[0] - 1]

            # add right pixels
            if mod != shape[0] - 1:
                sample_indices += [pixel - shape[0] + 1, pixel + 1, pixel + shape[0] + 1]

            # add top and bottom pixels
            sample_indices += [pixel - shape[0], pixel, pixel + shape[0]]

        # remove out of range values
        sample_indices = np.array(sample_indices)
        sample_indices = sample_indices[(sample_indices >= 0) & (sample_indices < shape[0] * shape[1])]

        # remove duplicates
        sample_indices = np.unique(sample_indices)

        indices.append(sample_indices)

    if flipping_method == "uniform_region":
        flipped_data = simple_sampling(batch, np.array(indices), "uniform")
    elif flipping_method == "gaussian_region":
        flipped_data = simple_sampling(batch, np.array(indices), "gaussian")
    else:
        raise ValueError("Region Perturbation method {} not known.".format(flipping_method))

    return flipped_data


def inpainting(batch, indicesfraction, flipping_method):
    flipped_data = []

    for s, sample in enumerate(batch):
        # build mask
        mask = np.zeros(sample.datum.shape[:2], dtype=np.uint8)
        np.put_along_axis(mask, indicesfraction[s], 1.0, axis=None)

        # get filepath
        # filepath = "not implemented"  # ToDo: how to get filepath/image file as expected?
        # img = cv2.imread(filepath, cv2.IMREAD_COLOR)

        datum = sample.datum

        datum = datum.astype(np.float32)

        if flipping_method == "inpaint_telea":
            # sample.filename
            for channel in range(datum.shape[2]):
                datum[:, :, channel] = cv2.inpaint(datum[:, :, channel], mask, 3, cv2.INPAINT_TELEA)
        elif flipping_method == "inpaint_ns":
            for channel in range(datum.shape[2]):
                datum[:, :, channel] = cv2.inpaint(datum[:, :, channel], mask, 3, cv2.INPAINT_NS)
        else:
            raise ValueError("Error in name of distribution to do inpainting. not implemented")

        flipped_data.append(datum)

    return flipped_data


FLIPPING_METHODS = {
    "uniform":  simple_sampling,
    "gaussian": simple_sampling,
    "uniform_region": region_perturbation,
    "gaussian_region": region_perturbation,
    "inpaint_telea": inpainting,
    "inpaint_ns": inpainting,
}


def load_explanations(explanationdir, samples, classidx):
    """ Load explanations for the given classidx. """

    explanations = []

    explanationdir = os.path.join(explanationdir, str(classidx))

    for sample in samples:
        explanations.append(np.load(os.path.join(explanationdir, extract_filename(sample.filename)) + ".npy"))

    return np.array(explanations)


def pixelflipping_wrapper(data_path, data_name, dataset_name, classidx, relevance_path, partition, batch_size, model_path, model_name, model_type, layer_name, rule, distribution, output_dir, percentage_values):
    """ Wrapper function to load data/model and compute directory paths. """

    n_neighbors = 20

    # construct explanationdir
    explanationdir = compute_relevance_path(relevance_path, data_name, model_name, "conv1", rule)
    explanationdir = os.path.join(explanationdir, partition)

    # init model
    model = init_model(model_path, model_name, framework=model_type)

    # load dataset for given class index
    datasetclass = get_dataset(dataset_name)
    class_data = datasetclass(data_path, "train", classidx=[classidx])
    class_data.set_mode("preprocessed")

    trainloader = DataLoader(class_data, batch_size=batch_size, endidx=300)

    X = []

    for batch in trainloader:
        data = np.array([b.datum for b in batch])
        activations = model.get_activations(data, layer_name)
        X.append([np.ravel(a) for a in activations])
        # X.append([np.ravel(b.datum) for b in batch])

    X = np.concatenate(X)

    print("Data shape is {}".format(X.shape))

    # train nearest neighbour estimator
    nbrs = NearestNeighbors(n_neighbors=n_neighbors)

    nbrs.fit(X)

    # load test data
    test_data = datasetclass(data_path, partition, classidx=[classidx])
    test_data.set_mode("preprocessed")

    testloader = DataLoader(test_data, batch_size=batch_size)

    # run nearest neighbor computation
    kappa_scores, gamma_scores, delta_scores, initial_kappa_scores, initial_gamma_scores, initial_delta_scores = \
        compute_nearest_neighbor_score(testloader, model, layer_name, nbrs, X, explanationdir, classidx, rule, distribution, percentage_values)

    # collect results and write to file
    results = []
    for key in kappa_scores:
        results.append([data_name, model_name, rule,
                        str(key), str(kappa_scores[key]), str(gamma_scores[key]), str(delta_scores[key]),
                        str(initial_kappa_scores[key]), str(initial_gamma_scores[key]), str(initial_delta_scores[key])])

    df = pd.DataFrame(results, columns=['dataset', 'model', 'method', 'flip_percentage', 'kappa', 'gamma', 'delta',
                                        'initial_kappa', 'initial_gamma', 'initial_delta'])

    os.makedirs(os.path.join(output_dir, layer_name), exist_ok=True)
    df.to_csv(
        os.path.join(output_dir, layer_name, "{}_{}_{}_{}_{}.csv".format(data_name, model_name, rule, distribution, str(classidx))),
        index=False)


def compute_nearest_neighbor_score(dataloader, model, layer_name, nbrs, X, explanationdir, classidx, rule, distribution, percentage_values):
    """ Estimate the pixelflipping score. """

    print("compute score for classidx {}".format(classidx))

    # save_examples = True
    #
    # if save_examples:
    #     examples_dir = os.path.join(output_dir, "examples", rule, str(classidx), distribution)
    #     if not os.path.exists(examples_dir):
    #         os.makedirs(examples_dir)

    distance_function = minkowski

    # prep result structure
    kappa_scores = {}
    gamma_scores = {}
    delta_scores = {}

    initial_kappa_scores = {}
    initial_gamma_scores = {}
    initial_delta_scores = {}

    for percentage in percentage_values:
        kappa_scores[percentage] = []
        gamma_scores[percentage] = []
        delta_scores[percentage] = []

        initial_kappa_scores[percentage] = []
        initial_gamma_scores[percentage] = []
        initial_delta_scores[percentage] = []

    # iterate data
    for b, batch in enumerate(dataloader):

        if rule != "random":
            # get/sort indices for pixelflipping order
            explanations = load_explanations(explanationdir, batch, classidx)
            # reduce explanations dimension
            explanations = np.max(explanations, axis=3)         # ToDo make compliant according to method

            if rule in ["Gradient", "SmoothGrad", "LRPZ"]:
                indices = [np.argsort(np.abs(explanation), axis=None) for explanation in explanations]
            else:
                indices = [np.argsort(explanation, axis=None) for explanation in explanations]

            indices = np.array(indices)

        else:
            # random
            indices = [np.argsort(np.max(sample.datum, axis=2), axis=None) for sample in batch]
            indices = np.array(indices)
            np.random.shuffle(indices)

        print(indices.shape)

        # loop flip_percentages
        for percentage in percentage_values:

            if percentage == 0:
                flipped_data = np.array([sample.datum for sample in batch])

            else:

                # get first percentage part of pixel indices (lowest relevance)
                # indicesfraction = indices[:, :int(flip_percentage * len(indices))]
                # get last percentage part of pixel indices (highest relevance)
                indicesfraction = indices[:, int((1 - percentage) * indices.shape[1]):]

                flipping_method = FLIPPING_METHODS[distribution]

                flipped_data = flipping_method(batch, indicesfraction, distribution)

            # get activations
            activations = model.get_activations(np.array(flipped_data), layer_name)
            flipped_data = np.array([np.ravel(a) for a in activations])
            # flipped_data = np.array([np.ravel(p) for p in flipped_data])

            # compute nearest neighbours
            nbr_distances, nbr_ind = nbrs.kneighbors(flipped_data)

            # save initial nearest neighbors
            if percentage == 0.0:
                initial_nbr_ind = nbr_ind
                initial_nbrs = X[initial_nbr_ind]

            # compute paper indices: Harmeling et al, 2006, From outliers to prototypes: Ordering data
            kappas = nbr_distances[:, -1]

            gammas = np.mean(nbr_distances, axis=1)

            deltas = []

            initial_kappas = []
            initial_gammas = []
            initial_deltas = []

            neighbor_points = X[nbr_ind]

            for p, point in enumerate(flipped_data):
                r_point = np.ravel(point)

                # compute initial kappas
                initial_kappas.append(distance_function(r_point, initial_nbrs[p][-1]))
                # compute initial gammas
                initial_gammas.append(np.mean(np.array([distance_function(r_point, initial_nbr) for initial_nbr in initial_nbrs[p]])))

                # deltas: compute length of mean of distance vectors to first k neighbours
                delta_vector = np.mean(np.array([r_point - neighbor_point for neighbor_point in neighbor_points[p]]), axis=0)
                # deltas for initial points
                initial_delta_vector = np.mean(np.array([r_point - initial_nbr for initial_nbr in initial_nbrs[p]]), axis=0)

                # append results
                deltas.append(distance_function(r_point, r_point + delta_vector))
                initial_deltas.append(distance_function(r_point, r_point + initial_delta_vector))


            # save examples
            # if save_examples and b == 0:
            #     for d, datapoint in enumerate(flipped_data[:10]):
            #         np.save(os.path.join(examples_dir, extract_filename(batch[d].filename)) + "_" + str(percentage) + ".npy", datapoint)

            kappa_scores[percentage].append(kappas)
            gamma_scores[percentage].append(gammas)
            delta_scores[percentage].append(np.array(deltas))

            initial_kappa_scores[percentage].append(np.array(initial_kappas))
            initial_gamma_scores[percentage].append(np.array(initial_gammas))
            initial_delta_scores[percentage].append(np.array(initial_deltas))

    for percentage in percentage_values:
        kappa_scores[percentage] = np.mean(np.concatenate(kappa_scores[percentage]))
        gamma_scores[percentage] = np.mean(np.concatenate(gamma_scores[percentage]))
        delta_scores[percentage] = np.mean(np.concatenate(delta_scores[percentage]))

        initial_kappa_scores[percentage] = np.mean(np.concatenate(initial_kappa_scores[percentage]))
        initial_gamma_scores[percentage] = np.mean(np.concatenate(initial_gamma_scores[percentage]))
        initial_delta_scores[percentage] = np.mean(np.concatenate(initial_delta_scores[percentage]))

    return kappa_scores, gamma_scores, delta_scores, initial_kappa_scores, initial_gamma_scores, initial_delta_scores


if __name__ == "__main__":

    def uncompress_percentages(string):
        percentages = string.split(":")
        percentages = [float(p) for p in percentages]
        return percentages

    # Setting up an argument parser for command line calls
    parser = argparse.ArgumentParser(description="Test and evaluate multiple xai methods")

    parser.add_argument("-d", "--data_path", type=str, default=None, help="data path")
    parser.add_argument("-dn", "--data_name", type=str, default=None, help="The name of the dataset to be used")
    parser.add_argument("-dl", "--dataloader_name", type=str, default=None, help="The name of the dataloader class to be used.")
    parser.add_argument("-rd", "--relevance_datapath", type=str, default=None, help="data folder of relevance maps")
    parser.add_argument("-o", "--output_dir", type=str, default="./output", help="Sets the output directory for the results")
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
    parser.add_argument("-pd", "--distribution", type=str, default="", help="Probability distribution to sample flipped pixels from (uniform, gaussian)")
    parser.add_argument("-pv", "--percentage_values", type=uncompress_percentages, help="Percentage values compressed as string value:value:value")

    ARGS = parser.parse_args()

    #####################
    #       MAIN
    #####################

    print("start pixelflipping now")
    start = time.process_time()
    tracemalloc.start()

    pixelflipping_wrapper(ARGS.data_path,
                          ARGS.data_name,
                          ARGS.dataloader_name,
                          ARGS.class_label,
                          ARGS.relevance_datapath,
                          ARGS.partition,
                          ARGS.batch_size,
                          ARGS.model_path,
                          ARGS.model_name,
                          ARGS.model_type,
                          ARGS.layer,
                          ARGS.rule,
                          ARGS.distribution,
                          ARGS.output_dir,
                          ARGS.percentage_values)

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    tracemalloc.stop()
    print("Duration of pixelflipping estimation:")
    print(time.process_time() - start)
