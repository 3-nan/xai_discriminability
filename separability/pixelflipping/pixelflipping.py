import argparse
import numpy as np
import tensorflow as tf
import pandas as pd

import innvestigate


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

flip_percentage = 0.3

# load data
if ARGS.data_path:
    print(ARGS.data_path)
else:
    print("load data from " + ARGS.data_name)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# preprocess data
x_train = x_train / 127.5 - 1
x_test = x_test / 127.5 - 1

# load model
if ARGS.model_path:
    # load model
    print(ARGS.model_path)
    model = tf.keras.models.load_model(ARGS.model_path)
else:
    # optionally add training sequence here??
    print("No model path given! Please add with -m model_path")

# estimate model score before flipping the pixels
pre_flip_score = model.evaluate(x_test, tf.one_hot(y_test, 10))

# preprocess model
model_wo_softmax = innvestigate.utils.keras.graph.model_wo_softmax(model)

# get analyzer
analyzer = parse_xai_method(ARGS.rule)
ana = analyzer(model_wo_softmax)

# compute relevance maps
rmaps = ana.analyze(x_test, neuron_selection=y_test, explained_layer_names=["conv2d"])

flipped_imgs = []

# iterate test data
for i, img in enumerate(x_test):
    rvalues = np.sum(rmaps["conv2d"][i], axis=2)

    # sort indices by relevance
    indices = np.argsort(rvalues, axis=None)

    # get first of pixel indices
    indices = indices[:(flip_percentage * len(indices))]

    # flip pixels
    for axis in range(img.shape[2]):
        uniform_values = np.random.uniform(-1.0, 1.0, len(indices))
        np.put_along_axis(img[:, :, axis], indices, uniform_values, axis=None)

    # save to array
    flipped_imgs.append(img)

flipped_imgs = np.array(flipped_imgs)

# estimate classification accuracy
flip_score = model.evaluate(flipped_imgs, tf.one_hot(y_test, 10))

# print results
print("estimated score before pixelflipping:")
print(pre_flip_score)
print("estimated score after pixelflipping:")
print(flip_score)

df = pd.DataFrame([[ARGS.data_name, ARGS.model_name, ARGS.rule, str(pre_flip_score), str(flip_score)]],
                  columns=['dataset', 'model', 'layer', 'method', 'actual score', 'flipped_score'])
df.to_csv(ARGS.output_dir + ARGS.data_name + "_" + ARGS.model_name + "_" + ARGS.rule + ".csv", index=False)
