import argparse
import os
import time
import tracemalloc
import copy
import numpy as np
import pandas as pd

from ..dataloading.custom import get_dataset
from ..dataloading.dataloader import DataLoader
from ..helpers.model_helper import init_model
from ..helpers.universal_helper import extract_filename, compute_relevance_path, join_path


# implement layer randomization
def layer_randomization(model, dataloader, classidx, xai_method, bottom_layer, explanationdir, output_dir,
                        top_down=True, independent=False):

    # configure save dir
    if not os.path.exists(join_path(output_dir, "explanations")):
        os.makedirs(join_path(output_dir, "explanations"))

    output_dir = join_path(output_dir, "explanations")

    # get layers including weights and iterate them
    layer_names = model.get_layer_names(model, with_weights_only=True)

    if top_down:
        layer_names = layer_names[::-1]

    results = {}

    for layer_name in layer_names:

        diff = []
        # randomize layer weights
        if independent:
            modelcp = copy.deepcopy(model)
            modelcp = modelcp.randomize_layer_weights(layer_name)
        else:
            model = model.randomize_layer_weights(layer_name)

        # iterate data and compute explanations
        for batch in dataloader:
            imgs = [sample.image for sample in batch]
            # labels = [sample.one_hot_label for sample in batch]

            if independent:
                explanations = modelcp.compute_relevance(imgs, bottom_layer, neuron_selection=classidx,
                                                         xai_method=xai_method, additional_parameter=None)
            else:
                explanations = model.compute_relevance(imgs, bottom_layer, neuron_selection=classidx,
                                                       xai_method=xai_method, additional_parameter=None)

            # save explanations and compute diff to original explanation

            for i, explanation in enumerate(explanations[bottom_layer]):

                np.save(join_path(output_dir, extract_filename(batch[i].filename)) + ".npy", explanation)

                # compute similarity
                original_explanation = np.load(join_path(explanationdir, ["val", classidx, batch[i].filename]) + ".npy")

                diff.append((np.square(original_explanation - explanation)).mean(axis=None))

        # compute results and save to dict
        diff = np.mean(diff)

        results[layer_name] = diff

    return results


def save_model_param_randomization_results(data_name, model_name, xai_method, classidx, class_results, outputdir):
    """ Save the results. """
    # save results
    results = []
    for layer in class_results:
        results.append([data_name, model_name, xai_method, str(classidx), layer, str(class_results[layer])])

    df = pd.DataFrame(results,
                      columns=['dataset', 'model', 'method', 'classindex', 'layer', 'score'])
    df.to_csv(join_path(outputdir, data_name + "_" + model_name + "_" + xai_method + ".csv"), index=False)


def model_parameter_randomization(data_path, data_name, dataset_name, partition, batch_size, model_path, model_name,
                                  bottom_layer, xai_method, explanationdir, output_dir):
    """ Function to create explanations on randomized models. """

    # init model
    model = init_model(model_path)

    # initialize dataset
    dataset_class = get_dataset(dataset_name)
    dataset = dataset_class(data_path, partition)

    explanationdir = compute_relevance_path(explanationdir, data_name, model_name, bottom_layer, xai_method)

    # configure directories
    if not os.path.exists(output_dir + "/" + "model_parameter_randomization"):
        os.makedirs(join_path(output_dir, "model_parameter_randomization"))
        os.makedirs(join_path(output_dir, "model_parameter_randomization/cascading_top_down"))
        os.makedirs(join_path(output_dir, "model_parameter_randomization/cascading_bottom_up"))
        os.makedirs(join_path(output_dir, "model_parameter_randomization/independent"))

    # iterate classes of the dataset
    for classname in dataset.classes:

        classidx = dataset.classname_to_idx(classname)
        print("iteration for class index {}".format(classidx))

        class_data = dataset_class(data_path, partition, classidx=[classidx])
        class_data = class_data.set_mode("preprocessed")

        print(type(class_data))

        dataloader = DataLoader(class_data, batch_size=batch_size)

        # CASE 1: cascading layer randomization top-down
        case_output_dir = join_path(output_dir, ["model_parameter_randomization", "cascading_top_down"])
        class_results = layer_randomization(model, dataloader, classidx, xai_method, bottom_layer,
                                            explanationdir, case_output_dir, top_down=True)
        # save results
        save_model_param_randomization_results(data_name, model_name, xai_method, classidx, class_results,
                                               case_output_dir)

        # CASE 2: cascading layer randomization bottom-up
        case_output_dir = join_path(output_dir, ["model_parameter_randomization", "cascading_bottom_up"])
        class_results = layer_randomization(model, dataloader, classidx, xai_method, bottom_layer,
                                            explanationdir, case_output_dir, top_down=False)
        # save results
        save_model_param_randomization_results(data_name, model_name, xai_method, classidx, class_results,
                                               case_output_dir)

        # CASE 3: independent layer randomization
        case_output_dir = join_path(output_dir, ["model_parameter_randomization", "independent"])
        class_results = layer_randomization(model, dataloader, classidx, xai_method, bottom_layer,
                                            explanationdir, case_output_dir, top_down=False, independent=True)
        # save results
        save_model_param_randomization_results(data_name, model_name, xai_method, classidx, class_results,
                                               case_output_dir)


if __name__ == "__main__":

    print("model parameter randomization")
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

    print("start relevance map computation now")
    start = time.process_time()
    tracemalloc.start()

    model_parameter_randomization(ARGS.data_path,
                                  ARGS.data_name,
                                  ARGS.dataset_name,
                                  ARGS.partition,
                                  ARGS.batch_size,
                                  ARGS.model_path,
                                  ARGS.model_name,
                                  ARGS.layer,
                                  ARGS.rule,
                                  ARGS.relevance_datapath,
                                  ARGS.output_dir
                                  )

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    tracemalloc.stop()
    print("Model Parameter randomization done")
    print("Duration of score computation:")
    print(time.process_time() - start)
    print("Job executed successfully.")
