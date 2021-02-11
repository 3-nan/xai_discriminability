import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from ..xaitestframework.helpers.universal_helper import join_path


filepath = "config_separability.yaml"

with open(filepath) as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)

    resultdir = "../results/one_class_separability"

    resultdir = resultdir + "/" + configs["data"]["dataname"] + "_" + configs["model"]["modelname"]

    layers = configs["layers"]
    classindices = range(20)
    print(layers)

    layer = layers[0]

    print(layer)

    method_scores = []

    plt.figure()

    for xai_method in configs["xai_methods"]:
        print(xai_method)
        scores = []
        for classidx in classindices:
            csv = pd.read_csv(resultdir + "/" + layer + "_" + xai_method + "_" + str(classidx) + ".csv")
            # print(float(csv["separability_score"]))
            scores.append(float(csv["separability_score"]))

        method_scores.append(np.mean(scores))

        plt.plot(range(20), scores)

    plt.xlabel("class_label")
    plt.ylabel("separability score")
    plt.legend(configs["xai_methods"])
    plt.show()

    plt.figure()
    plt.bar(configs["xai_methods"], method_scores)
    plt.xlabel("xai method")
    plt.ylabel("separability score [mean over classes]")
    plt.xticks(rotation="45", ha="right")
    plt.show()

    plt.figure()

    for xai_method in configs["xai_methods"]:
        scores = []

        for layer in layers:

            layer_scores = []

            for classidx in classindices:
                try:
                    csv = pd.read_csv(resultdir + "/" + layer + "_" + xai_method + "_" + str(classidx) + ".csv")
                    layer_scores.append(float(csv["separability_score"]))
                except FileNotFoundError:
                    print(resultdir + "/" + layer + "_" + xai_method + "_" + str(classidx) + ".csv not found")
                # print(float(csv["separability_score"]))

            scores.append(np.mean(layer_scores))

        plt.plot(layers, scores)

    plt.xlabel("xai method")
    plt.ylabel("class-wise mean of separability scores")
    plt.legend(configs["xai_methods"])
    plt.show()
