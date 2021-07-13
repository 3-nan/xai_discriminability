import os
import argparse
import time
import numpy as np
import tracemalloc
import cv2
from openTSNE import TSNE, TSNEEmbedding
from openTSNE import initialization
from openTSNE.affinity import PerplexityBasedNN
# from openTSNE.callbacks import ErrorLogger

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

        datum = sample.datum.astype(np.float32)

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


def flip_data(dataloader, explanationdir, classidx, model_path, model_name, model_type, layer_name, rule, distribution, percentage):
    """ Estimate the pixelflipping score. """

    print("compute score for classidx {}".format(classidx))

    model = init_model(model_path, model_name, model_type)

    X_flipped = []

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

        if percentage == 0:
            flipped_data = np.array([sample.datum for sample in batch])

        else:

            # get first percentage part of pixel indices (lowest relevance)
            # indicesfraction = indices[:, :int(flip_percentage * len(indices))]
            # get last percentage part of pixel indices (highest relevance)
            indicesfraction = indices[:, int((1 - percentage) * indices.shape[1]):]

            flipping_method = FLIPPING_METHODS[distribution]

            flipped_data = flipping_method(batch, indicesfraction, distribution)

        # ravel
        activations = model.get_activations(np.array(flipped_data), layer_name)
        flipped_data = np.array([np.ravel(a) for a in activations])

        # embed flipped data
        X_flipped.append(flipped_data)

    return np.concatenate(X_flipped)


def tsne_embedding_evaluation(data_path, data_name, dataset_name, explanationdir, partition, batch_size, model_path, model_name, model_type, layer_name, rule, distribution, output_dir, percentage_values, add_points=True):

    os.makedirs(os.path.join(output_dir, layer_name, distribution), exist_ok=True)
    output_dir = os.path.join(output_dir, layer_name, distribution)
    # compute explanation dir
    explanationdir = compute_relevance_path(explanationdir, data_name, model_name, "conv1", rule)
    explanationdir = os.path.join(explanationdir, partition)

    # compute tsne embedding
    print("Computing tsne embedding for train data.")
    datasetclass = get_dataset(dataset_name)

    # class_indices = [str(data.classname_to_idx(name)) for name in data.classes]

    idx = 15

    # load model
    model = init_model(model_path, model_name, framework=model_type)

    # load data for tsne embedding computation
    X = []
    targets = []

    class_data = datasetclass(data_path, "train", classidx=[idx])
    class_data.set_mode("preprocessed")
    trainloader = DataLoader(class_data, batch_size=batch_size)  # , endidx=150)

    for batch in trainloader:
        data = np.array([b.datum for b in batch])
        activations = model.get_activations(data, layer_name)
        X.append([np.ravel(a) for a in activations])

    targets = np.ones(len(trainloader.idx))  # * int(idx)

    # add flipped data
    test_data = datasetclass(data_path, "val", classidx=[idx])
    test_data.set_mode("preprocessed")
    test_loader = DataLoader(test_data, batch_size=batch_size)

    for p, percentage in enumerate(percentage_values):
        X_flipped = flip_data(test_loader, explanationdir, idx, model_path, model_name, model_type, layer_name, rule, distribution, percentage)
        X.append(X_flipped)
        targets = np.concatenate((targets, np.ones(X_flipped.shape[0]) * (p + 1)))

    X = np.concatenate(X)
    targets = np.array(targets)

    print("Data shape is {}".format(X.shape))

    # initialize tsne embedding
    affinities_train = PerplexityBasedNN(
        X,
        perplexity=30,
        method="exact",
        # metric="euclidean",
        n_jobs=8,
        random_state=42,
    )

    init_train = initialization.pca(X, random_state=42)

    tsne = TSNEEmbedding(
        init_train,
        affinities_train,
        negative_gradient_method="fft",
        n_jobs=8,
        # perplexity=30,
        # metric="euclidean",
        # callbacks=ErrorLogger(),
        # n_jobs=8,
        # random_state=42
    )

    # fit tsne embedding
    # tsne_embedding = tsne.fit(X)
    embedding_train_1 = tsne.optimize(n_iter=500, exaggeration=1, momentum=0.8, inplace=True)
    # embedding_train = tsne.optimize(n_iter=250, exaggeration=12, momentum=0.5, inplace=True)
    # embedding_train_1 = embedding_train.optimize(n_iter=750, exaggeration=1, momentum=0.8, inplace=True)

    # embedding_train_1 = embedding_train.optimize(n_iter=500, exaggeration=1, momentum=0.8, inplace=True)

    # save embedding
    np.save(os.path.join(output_dir, "tsne_embedding.npy"), embedding_train_1)  # Todo add path
    np.save(os.path.join(output_dir, "targets.npy"), targets)
    print("t-SNE embedding computed.")



# def tsne_embedding_evaluation(data_path, data_name, dataset_name, explanationdir, partition, batch_size, model_path, model_name, model_type, layer_name, rule, distribution, output_dir, percentage_values, add_points=True):
#
#     # compute explanation dir
#     explanationdir = compute_relevance_path(explanationdir, data_name, model_name, "conv1", rule)
#     explanationdir = os.path.join(explanationdir, partition)
#
#     # compute tsne embedding
#     print("Computing tsne embedding for train data.")
#     tsne_embedding = compute_tsne_embedding(data_path, dataset_name, "train", batch_size, model_path, model_name, model_type, layer_name, output_dir)
#     print("t-SNE embedding computed.")
#
#     if add_points:
#
#         datasetclass = get_dataset(dataset_name)
#
#         idx = 15
#
#         test_data = datasetclass(data_path, partition, classidx=[idx])
#         test_data.set_mode("preprocessed")
#         testloader = DataLoader(test_data, batch_size=batch_size)
#
#         transform_to_tsne_embedding(testloader, idx, tsne_embedding, explanationdir, model_path, model_name, model_type, layer_name, rule, distribution,
#                                     percentage_values, output_dir)


def compute_tsne_embedding(data_path, dataset_name, partition, batch_size, model_path, model_name, model_type, layer_name, output_dir):
    """ Wrapper function to load data/model and compute directory paths. """

    # load dataset for given class index
    datasetclass = get_dataset(dataset_name)
    data = datasetclass(data_path, partition)

    # class_indices = [str(data.classname_to_idx(name)) for name in data.classes]

    idx = 15

    # load model
    model = init_model(model_path, model_name, framework=model_type)

    # load data for tsne embedding computation
    X = []
    targets = []

    class_data = datasetclass(data_path, partition, classidx=[idx])
    class_data.set_mode("preprocessed")
    trainloader = DataLoader(class_data, batch_size=batch_size) # , endidx=150)

    for batch in trainloader:
        data = np.array([b.datum for b in batch])
        activations = model.get_activations(data, layer_name)
        X.append([np.ravel(a) for a in activations])

    targets = np.ones(len(trainloader.idx)) #   * int(idx)

    X = np.concatenate(X)
    targets = np.array(targets)

    print("Data shape is {}".format(X.shape))

    # initialize tsne embedding
    affinities_train = PerplexityBasedNN(
        X,
        perplexity=6,
        method="approx",
        # metric="euclidean",
        n_jobs=8,
        random_state=42,
    )

    init_train = initialization.pca(X, random_state=42)

    tsne = TSNEEmbedding(
        init_train,
        affinities_train,
        negative_gradient_method="fft",
        n_jobs=8,
        # perplexity=30,
        # metric="euclidean",
        # callbacks=ErrorLogger(),
        # n_jobs=8,
        # random_state=42
    )

    # fit tsne embedding
    # tsne_embedding = tsne.fit(X)
    embedding_train_1 = tsne.optimize(n_iter=500, exaggeration=1, momentum=0.8, inplace=True)
    # embedding_train = tsne.optimize(n_iter=250, exaggeration=12, momentum=0.5, inplace=True)
    # embedding_train_1 = embedding_train.optimize(n_iter=750, exaggeration=1, momentum=0.8, inplace=True)

    # embedding_train_1 = embedding_train.optimize(n_iter=500, exaggeration=1, momentum=0.8, inplace=True)

    # save embedding
    np.save(os.path.join(output_dir, "tsne_embedding_layer_6.npy"), embedding_train_1)     # Todo add path
    np.save(os.path.join(output_dir, "targets_layer_6.npy"), targets)

    return embedding_train_1


def transform_to_tsne_embedding(testloader, classidx, tsne_embedding, explanationdir, model_path, model_name, model_type, layer_name, rule, distribution, percentage_values, outputdir):

    print(percentage_values)
    # fit new datapoints to existing embedding
    for percentage in percentage_values:
        X_embedded = transform_to_embedding(testloader, tsne_embedding, explanationdir, classidx, model_path, model_name, model_type, layer_name, rule, distribution, percentage)

        os.makedirs(os.path.join(outputdir, str(classidx)), exist_ok=True)
        np.save(os.path.join(outputdir, str(classidx), "percentage_{}.npy".format(percentage)), X_embedded)     # Todo add path


def transform_to_embedding(dataloader, tsne_embedding, explanationdir, classidx, model_path, model_name, model_type, layer_name, rule, distribution, percentage):
    """ Estimate the pixelflipping score. """

    print("compute score for classidx {}".format(classidx))

    model = init_model(model_path, model_name, model_type)

    X_embedded = []

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

        if percentage == 0:
            flipped_data = np.array([sample.datum for sample in batch])

        else:

            # get first percentage part of pixel indices (lowest relevance)
            # indicesfraction = indices[:, :int(flip_percentage * len(indices))]
            # get last percentage part of pixel indices (highest relevance)
            indicesfraction = indices[:, int((1 - percentage) * indices.shape[1]):]

            flipping_method = FLIPPING_METHODS[distribution]

            flipped_data = flipping_method(batch, indicesfraction, distribution)

        # ravel
        activations = model.get_activations(np.array(flipped_data), layer_name)
        flipped_data = np.array([np.ravel(a) for a in activations])

        # embed flipped data
        X_embedded.append(tsne_embedding.transform(flipped_data))

    return X_embedded


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

    print("start tsne evaluation now")
    start = time.process_time()
    tracemalloc.start()

    tsne_embedding_evaluation(ARGS.data_path,
                              ARGS.data_name,
                              ARGS.dataloader_name,
                              # ARGS.class_label,
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
    print("Duration of tsne manifold estimation:")
    print(time.process_time() - start)
