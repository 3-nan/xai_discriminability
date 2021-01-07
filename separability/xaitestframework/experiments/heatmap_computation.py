import argparse
import datetime
import time
import os
import numpy as np

from ..dataloading.dataloader import get_dataloader
from ..helpers.model_helper import init_model
from ..helpers.universal_helper import extract_filename


def get_relevance_dir_path(output_dir, data_name, model_name, layer, rule, partition, class_name):
    """ Computes directory path to save computed relevance to. """

    # remove backslash at the end
    if output_dir[-1] == "/":
        output_dir = output_dir[:-1]

    for attr in [data_name, model_name, layer, rule, partition, str(class_name)]:

        if not os.path.exists(output_dir + "/" + attr):
            os.makedirs(output_dir + "/" + attr)
        output_dir = output_dir + "/" + attr

    return output_dir


def compute_relevances_for_class(data_path, data_name, dataloader_name, partition, batch_size, startidx, endidx, model_path, model_name, layer_names, xai_method, class_name, output_dir):
    """ Function to compute the attributed relevances for the selected class. """

    # init model
    model = init_model(model_path)

    # initialize dataloader
    dataloader = get_dataloader(dataloader_name)
    dataloader = dataloader(datapath=data_path, partition=partition, batch_size=batch_size)

    # get dataset partition
    data, labels = dataloader.get_dataset_partition(startidx=startidx, endidx=endidx, batched=True)

    for i, batch in enumerate(data):

        print("compute relevances for batch {}".format(i))
        preprocessed, _ = dataloader.preprocess_data(batch, labels[i])

        # compute relevance
        R = model.compute_relevance(preprocessed, layer_names, class_name, xai_method, additional_parameter=None)

        for layer_name in layer_names:
            layer_output_dir = get_relevance_dir_path(output_dir, data_name, model_name, layer_name, xai_method, partition, class_name)
            for r, relevance in enumerate(R[layer_name]):
                fname = extract_filename(batch[r])
                filename = layer_output_dir + "/" + fname + ".npy"
                np.save(filename, relevance)


current_datetime = datetime.datetime.now()
print(current_datetime)


def decode_layernames(string):
    """ Decodes the layer_names string to a list of strings. """
    return string.split(":")


# Setting up an argument parser for command line calls
parser = argparse.ArgumentParser(description="Test and evaluate multiple xai methods")

parser.add_argument("-d", "--data_path", type=str, default=None, help="data path")
parser.add_argument("-dn", "--data_name", type=str, default=None, help="The name of the dataset to be used")
parser.add_argument("-dl", "--dataloader_name", type=str, default=None, help="The name of the dataloader class to be used.")
parser.add_argument("-o", "--output_dir", type=str, default="./output", help="Sets the output directory for the results")
parser.add_argument("-m", "--model_path", type=str, default=None, help="path to the model")
parser.add_argument("-mn", "--model_name", type=str, default=None, help="Name of the model to be used")
parser.add_argument("-si", "--start_index", type=int, default=0, help="Index of dataset to start with")
parser.add_argument("-ei", "--end_index", type=int, default=50000, help="Index of dataset to end with")
parser.add_argument("-p", "--partition", type=str, default="train", help="Either train or test for one of these partitions")
parser.add_argument("-cl", "--class_label", type=int, default=0, help="Index of class to compute heatmaps for")
parser.add_argument("-r", "--rule", type=str, default="LRPSequentialCompositeA", help="Rule to be used to compute relevance maps")
parser.add_argument("-l", "--layer_names", type=decode_layernames, default=None, help="Layer to compute relevance maps for")
parser.add_argument("-bs", "--batch_size", type=int, default=50, help="Batch size for relevance map computation")

ARGS = parser.parse_args()

#####################
#       MAIN
#####################

print("start relevance map computation now")
start = time.process_time()

compute_relevances_for_class(ARGS.data_path,
                             ARGS.data_name,
                             ARGS.dataloader_name,
                             ARGS.partition,
                             ARGS.batch_size,
                             ARGS.start_index,
                             ARGS.end_index,
                             ARGS.model_path,
                             ARGS.model_name,
                             ARGS.layer_names,
                             ARGS.rule,
                             ARGS.class_label,
                             ARGS.output_dir)


print("Relevance maps for x_data computed")
print("Duration of relevance map computation:")
print(time.process_time() - start)
