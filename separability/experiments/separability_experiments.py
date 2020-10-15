import argparse
import datetime
import time
import numpy as np
import tensorflow as tf

import innvestigate

current_datetime = datetime.datetime.now()
print(current_datetime)

# Setting up an argument parser for command line calls
parser = argparse.ArgumentParser(description="Test and evaluate multiple xai methods")

parser.add_argument("-d", "--data_path", type=str, default=None, help="data path")
parser.add_argument("-dn", "--data_name", type=str, default=None, help="The name of the dataset to be used")
parser.add_argument("-o", "--output_dir", type=str, default="./output", help="Sets the output directory for the results")
parser.add_argument("-m", "--model_path", type=str, default=None, help="path to the model")
parser.add_argument("-mn", "--model_name", type=str, default=None, help="Name of the model to be used")

ARGS = parser.parse_args()

#####################
#       MAIN
#####################

# load data
if ARGS.data_path:
    print(ARGS.data_path)
else:
    print("load data from " + ARGS.data_name)

# prepare model
model_path = ARGS.model_path
if model_path:
    # load model
    print(model_path)
    model = tf.keras.models.load_model(model_path)
else:
    # optionally add training sequence here??
    print("No model path given! Please add with -m model_path")

# execute experiments
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# preprocess data
x_test = x_test / 127.5 - 1

# convert to one hot encoding
y_test_onehot = tf.one_hot(y_test, 10)[:, 0, :]

score = model.evaluate(x_test, y_test_onehot)
print(score)

print("start relevance map computation now")
    # compute relevance maps
    # save relevance maps, optionally do that in computation module

start = time.process_time()

model = innvestigate.utils.keras.graph.model_wo_softmax(model)

analyzer = innvestigate.analyzer.LRPSequentialCompositeA

ana = analyzer(model)

R = []
layer = "conv2d"
neuron_selection = np.zeros(200)    # "index"

for batch in range(int(x_test.shape[0]/200.)-40):
    R_batch = ana.analyze(x_test[batch*200:(batch+1)*200],
                          neuron_selection=neuron_selection,
                          explained_layer_names=[layer])
    R.append(np.array(R_batch[layer]))

R = np.concatenate(R)

print("Relevance maps for x_test computed")
print("Duration of relevance map computation:")
print(time.process_time() - start)
