import argparse
import os
import numpy as np
import tensorflow as tf
import pandas as pd

from . import innvestigate
from ..dataloading import Dataloader


def load_relevance_maps_for_label(data_path, partition, label):
    """ Load a selection of relevance maps
    label: int - class label to get data for
    """

    dir_path = data_path + "/" + partition + "/" + str(label)

    # load data from indices
    data_files = os.listdir(dir_path)
    data_files.sort(key=lambda f: int(f.split("_", 1)[0]))

    R_c = []
    for file in data_files:
        R_c.append(np.load(dir_path + "/" + file))

    R_c = np.concatenate(R_c)

    return R_c


def parse_xai_method(xai_method):
    # Gradient methods
    if xai_method == "Gradient":
        analyzer = innvestigate.analyzer.Gradient
    elif xai_method == "SmoothGrad":
        analyzer = innvestigate.analyzer.SmoothGrad
    # LRP methods
    elif xai_method == "LRPZ":
        analyzer = innvestigate.analyzer.LRPZ
    elif xai_method == 'LRPEpsilon':
        analyzer = innvestigate.analyzer.LRPEpsilon
    elif xai_method == 'LRPWSquare':
        analyzer = innvestigate.analyzer.LRPWSquare
    elif xai_method == 'LRPGamma':
        analyzer = innvestigate.analyzer.LRPGamma
    elif xai_method == 'LRPAlpha1Beta0':
        analyzer = innvestigate.analyzer.LRPAlpha1Beta0
    elif xai_method == 'LRPAlpha2Beta1':
        analyzer = innvestigate.analyzer.LRPAlpha2Beta1
    elif xai_method == 'LRPSequentialPresetA':
        analyzer = innvestigate.analyzer.LRPSequentialPresetA
    elif xai_method == 'LRPSequentialPresetB':
        analyzer = innvestigate.analyzer.LRPSequentialPresetB
    elif xai_method == 'LRPSequentialCompositeA':
        analyzer = innvestigate.analyzer.LRPSequentialCompositeA
    elif xai_method == 'LRPSequentialCompositeB':
        analyzer = innvestigate.analyzer.LRPSequentialCompositeB
    elif xai_method == 'LRPSequentialCompositeBFlat':
        analyzer = innvestigate.analyzer.LRPSequentialCompositeBFlat
    # innvestigate.analyzer.LRPGamma
    else:
        print("analyzer name " + xai_method + " not correct")
        analyzer = None
    return analyzer


# Setting up an argument parser for command line calls
parser = argparse.ArgumentParser(description="Test and evaluate multiple xai methods")

parser.add_argument("-d", "--data_path", type=str, default=None, help="data path")
parser.add_argument("-dn", "--data_name", type=str, default=None, help="The name of the dataset to be used")
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

flip_percentage = 0.5
flip_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# load data
# if ARGS.data_path:
#     print(ARGS.data_path)
# else:
#     print("load data from " + ARGS.data_name)
#     (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# preprocess data
# x_train = x_train / 127.5 - 1
# x_test = x_test / 127.5 - 1

dataloader = Dataloader(datapath=ARGS.data_path, batch_size=ARGS.batch_size)
data = dataloader.get_data("val")
y_test, classes = dataloader.get_indices("val")

# load model
if ARGS.model_path:
    # load model
    print(ARGS.model_path)
    model = tf.keras.models.load_model(ARGS.model_path)
else:
    # optionally add training sequence here??
    print("No model path given! Please add with -m model_path")

# estimate model score before flipping the pixels
pre_flip_score = model.evaluate(data, verbose=0)

# preprocess model
# model_wo_softmax = innvestigate.utils.keras.graph.model_wo_softmax(model)
#
# # get analyzer
# analyzer = parse_xai_method(ARGS.rule)
# ana = analyzer(model_wo_softmax)
#
# # compute relevance maps
# rmaps = ana.analyze(x_test, neuron_selection=y_test, explained_layer_names=["conv2d"])

imgs = []
R_maps = []
all_Rs = []

# load relevance maps
for c in classes:

    R_c = load_relevance_maps_for_label(ARGS.relevance_datapath + "/" + ARGS.data_name + "/" + ARGS.model_name + "/" + ARGS.layer + "/" + ARGS.rule + "/", "val", c)
    all_Rs.append(R_c)

print(y_test.shape)
print(len(all_Rs))
# filter correct relevance maps
for i, label in enumerate(y_test):
    index = np.where(label == 1)[0][0]
    index = list(classes).index(index)

    R_maps.append(all_Rs[index][i])
R_maps = np.array(R_maps)
print(R_maps.shape)

# load images to array
for batch in data.as_numpy_iterator():
    imgs.append(batch[0])
imgs = np.concatenate(imgs)
print("img to array done")
print(imgs.shape)

assert (R_maps.shape == imgs.shape)
print("assert statement passed")

# order image indices by relevance
R_maps = np.sum(R_maps, axis=3)

# iterate flip percentage values
results = []

for flip_percentage in flip_percentages:
    flipped_imgs = []

    # iterate test data
    for i, img in enumerate(imgs):

        # sort indices by relevance
        indices = np.argsort(R_maps[i], axis=None)

        print(np.take_along_axis(R_maps[i], indices))
        # get first of pixel indices
        # indices = indices[:int(flip_percentage * len(indices))]
        # get last of pixel indices
        indices = indices[int(flip_percentage * len(indices)):]

        # flip pixels
        for axis in range(img.shape[2]):
            uniform_values = np.random.uniform(-1.0, 1.0, len(indices))
            np.put_along_axis(img[:, :, axis], indices, uniform_values, axis=None)

        # save to array
        flipped_imgs.append(img)

    flipped_imgs = np.array(flipped_imgs)
    print("images flipped")

    # estimate classification accuracy
    flip_score = model.evaluate(flipped_imgs, y_test)

    # print results
    print("estimated score before pixelflipping:")
    print(pre_flip_score)
    print("estimated score after pixelflipping:")
    print(flip_score)

    results.append([ARGS.data_name, ARGS.model_name, ARGS.rule, str(flip_percentage), str(pre_flip_score), str(flip_score)])

    df = pd.DataFrame(results,
                      columns=['dataset', 'model', 'method', 'flip_percentage', 'actual score', 'flipped_score'])
    df.to_csv(ARGS.output_dir + ARGS.data_name + "_" + ARGS.model_name + "_" + ARGS.rule + ".csv", index=False)
