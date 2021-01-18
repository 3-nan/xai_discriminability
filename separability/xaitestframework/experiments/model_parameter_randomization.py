import argparse
import os
import numpy as np
import tensorflow as tf

from . import innvestigate
from ..dataloading.dataloader import DataLoader
from ..helpers.analyzer_helper import parse_xai_method


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


def model_parameter_randomization():
    """ Function to create explanations on randomized models. """
    # init model
    model = init_model(model_path)

    # initialize dataloader
    dataloader = get_dataloader(dataloader_name)
    dataloader = dataloader(datapath=data_path, partition=partition, batch_size=batch_size)

    # preprocess data
    test_data, test_labels = dataloader.preprocess_data(dataloader.samples, dataloader.labels)

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


# CASE 1: cascading randomization
def cascading_layer_randomization(model, analyzer, bottom_layer, x_test, y_test, output_dir, top_down=True):
    # iterate layers
    layer_gen = [layer for layer in model.layers if hasattr(layer, 'kernel_initializer')]

    if top_down:
        layer_gen = layer_gen[::-1]

    for layer in layer_gen:
        # randomize layer weights
        weights = layer.get_weights()
        weight_initializer = layer.kernel_initializer
        model.get_layer(name=layer.name).set_weights(
            [weight_initializer(shape=weights[0].shape), weights[1]])

        # get analyzer
        ana = analyzer(model)

        # compute relevance maps
        rmaps = ana.analyze(x_test, neuron_selection=y_test, explained_layer_names=[bottom_layer])

        rmaps = np.array(rmaps[ARGS.layer])

        if not top_down:
            if not os.path.exists(output_dir + "bottom_up/"):
                os.makedirs(output_dir + "bottom_up/")
            np.save(output_dir + "bottom_up/" + layer.name + ".npy", rmaps)

        else:
            np.save(output_dir + layer.name + ".npy", rmaps)


# CASE 2: independent randomization
def independent_layer_randomization(model, analyzer, bottom_layer, x_test, y_test, output_dir):
    # iterate layers
    layer_gen = (layer for layer in model.layers if hasattr(layer, 'kernel_initializer'))

    for layer in layer_gen:
        cloned_model = tf.keras.models.clone_model(model)
        # randomize layer weights
        weights = layer.get_weights()
        weight_initializer = layer.kernel_initializer
        cloned_model.get_layer(name=layer.name).set_weights(
            [weight_initializer(shape=weights[0].shape), weights[1]])

        # get analyzer
        ana = analyzer(cloned_model)

        # compute relevance maps
        rmaps = ana.analyze(x_test, neuron_selection=y_test, explained_layer_names=[bottom_layer])

        rmaps = np.array(rmaps[ARGS.layer])

        np.save(output_dir + layer.name + ".npy", rmaps)


# MAIN FUNCTION
dataloader = DataLoader(datapath=ARGS.data_path, batch_size=ARGS.batch_size)
data = dataloader.get_data("val", shuffle=True, seed=42)
datapartition = data.take(1)
x_test = []
y_test = []
for dp in datapartition.as_numpy_iterator():
    x_test.append(np.array(dp[0]))
    y_test.append(np.array(dp[1]))

x_test = np.array(x_test)[0]
y_test = np.array(y_test)[0]

y_test = np.argmax(y_test, axis=1)
print(y_test.shape)
print(x_test.shape)

# make output dirs
output_dir = ARGS.output_dir + ARGS.data_name + "_" + ARGS.model_name + "_" + ARGS.rule
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    os.makedirs(output_dir + "/" + "cascading")
    os.makedirs(output_dir + "/" + "independent")

np.save(output_dir + "/" + "original" + ".npy", np.array(x_test))

# load model
if ARGS.model_path:
    # load model
    print(ARGS.model_path)
    model = tf.keras.models.load_model(ARGS.model_path)
else:
    # optionally add training sequence here??
    print("No model path given! Please add with -m model_path")


# preprocess model
model = innvestigate.utils.keras.graph.model_wo_softmax(model)

# get analyzer
analyzer = parse_xai_method(ARGS.rule)

# get analyzer and compute normal relevance maps
ana = analyzer(model)

# compute relevance maps
rmaps = ana.analyze(x_test, neuron_selection=y_test, explained_layer_names=[ARGS.layer])
rmaps = np.array(rmaps[ARGS.layer])

np.save(output_dir + "/" + "not_randomized" + ".npy", rmaps)

print("cascading layer randomization")
cascading_layer_randomization(model, analyzer, ARGS.layer, x_test, y_test, output_dir + "/cascading/")
print("cascading layer randomization bottom up")
cascading_layer_randomization(model, analyzer, ARGS.layer, x_test, y_test, output_dir + "/cascading/", top_down=False)
print("independent layer randomization")
independent_layer_randomization(model, analyzer, ARGS.layer, x_test, y_test, output_dir + "/independent/")