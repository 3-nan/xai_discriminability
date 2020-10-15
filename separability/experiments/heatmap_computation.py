import argparse
import datetime
import time
import os
import numpy as np
import tensorflow as tf

import innvestigate


def parse_xai_method(xai_method):
    # Gradient methods
    if xai_method == "Gradient":
        analyzer = innvestigate.analyzer.Gradient
    # LRP methods
    elif xai_method == "LRPZ":
        analyzer = innvestigate.analyzer.LRPZ
    elif xai_method == 'LRPEpsilon':
        analyzer = innvestigate.analyzer.LRPEpsilon
    elif xai_method == 'LRPWSquare':
        analyzer = innvestigate.analyzer.LRPWSquare
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


current_datetime = datetime.datetime.now()
print(current_datetime)

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

# load data
if ARGS.data_path:
    print(ARGS.data_path)
else:
    print("load data from " + ARGS.data_name)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# preprocess data
x_train = x_train / 127.5 - 1
x_test = x_test / 127.5 - 1

# get data selection
x_data = x_train[ARGS.start_index:ARGS.end_index]

# prepare model
model_path = ARGS.model_path
if model_path:
    # load model
    print(model_path)
    model = tf.keras.models.load_model(model_path)
else:
    # optionally add training sequence here??
    print("No model path given! Please add with -m model_path")


print("start relevance map computation now")
start = time.process_time()

model = innvestigate.utils.keras.graph.model_wo_softmax(model)

# get analyzer
analyzer = parse_xai_method(ARGS.rule)

ana = analyzer(model)

R = []
layer = ARGS.layer
neuron_selection = ARGS.class_label # np.ones(ARGS.batch_size) * ARGS.class_label    # "index"

for batch in range(int(x_data.shape[0]/ARGS.batch_size)):
    R_batch = ana.analyze(x_data[batch*ARGS.batch_size:(batch+1)*ARGS.batch_size],
                          neuron_selection=neuron_selection,
                          explained_layer_names=[layer])
    R.append(np.array(R_batch[layer]))

R = np.concatenate(R)

print("Relevance maps for x_data computed")
print("Duration of relevance map computation:")
print(time.process_time() - start)

# save relevance maps
# /data/cluster/users/motzkus/relevance_maps/
output_dir = ARGS.output_dir

for attr in [ARGS.data_name, ARGS.model_name, ARGS.layer, ARGS.rule, ARGS.partition, str(ARGS.class_label)]:

    if not os.path.exists(output_dir + "/" + attr):
        os.makedirs(output_dir + "/" + attr)
    output_dir = output_dir + "/" + attr

np.save(output_dir + "/" + str(ARGS.start_index) + "_" + str(ARGS.end_index) + ".npy", R)
