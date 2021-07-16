import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from ..xaitestframework.helpers.universal_helper import join_path


filepath = "configs/config_experiments.yaml"

with open(filepath) as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)

    resultdir = "../results/separability"

    resultdir = resultdir + "/" + configs["data"]["dataname"] + "_" + configs["model"]["modelname"]

    layers = configs["layers"]

    classindices = range(20)
    print(layers)

    layer = layers[0]

    print(layer)

    method_scores = []

    # plt.figure()
    #
    # for xai_method in configs["xai_methods"]:
    #     print(xai_method)
    #     scores = []
    #     for classidx in classindices:
    #         csv = pd.read_csv(resultdir + "/" + layer + "_" + xai_method + "_" + str(classidx) + ".csv")
    #         # print(float(csv["separability_score"]))
    #         scores.append(float(csv["separability_score"]))
    #
    #     method_scores.append(np.mean(scores))
    #
    #     plt.plot(range(20), scores)
    #
    # plt.xlabel("class_label")
    # plt.ylabel("separability score")
    # plt.legend(configs["xai_methods"])
    # plt.show()

    # plt.figure()
    # plt.bar(configs["xai_methods"], method_scores)
    # plt.xlabel("xai method")
    # plt.ylabel("separability score [mean over classes]")
    # plt.xticks(rotation="45", ha="right")
    # plt.show()

    # plt.figure()

    # fig, ax = plt.subplots()

    x = np.arange(len(layers))
    n_classes = len(classindices)

    width = 0.8
    cmap = plt.cm.get_cmap("Set1", len(configs["xai_methods"]))

    # for m, xai_method in enumerate(configs["xai_methods"]):
    #     scores = []
    #
    #     for c, classidx in enumerate(classindices):
    #
    #         class_scores = []
    #
    #         for layer in layers:
    #
    #             try:
    #                 csv = pd.read_csv(resultdir + "/" + layer + "_" + xai_method + "_" + str(classidx) + ".csv")
    #                 class_scores.append(float(csv["separability_score"]))
    #             except FileNotFoundError:
    #                 print(resultdir + "/" + layer + "_" + xai_method + "_" + str(classidx) + ".csv not found")
    #                 class_scores.append(0.)
    #             # print(float(csv["separability_score"]))
    #
    #         # print(len(layers))
    #         # print(len(class_scores))
    #         ax.bar(x - width / 2. + c * (width/n_classes), class_scores, width / n_classes, label=classidx, color=cmap(m), alpha=0.5)
    #
    #     # plt.plot(layers, scores)
    #
    # handles, labels = ax.get_legend_handles_labels()
    #
    # plt.xticks(x, layers)
    # plt.xlabel("Layer")
    # plt.ylabel("class-wise mean of margins")
    # plt.legend(handles[::n_classes], configs["xai_methods"])
    # plt.show()

    plt.figure()

    for m, xai_method in enumerate(configs["xai_methods"]):
        scores = []

        for layer in layers:

            layer_scores = []

            for c, classidx in enumerate(classindices):

                try:
                    csv = pd.read_csv(resultdir + "/" + layer + "_" + xai_method + "_" + str(classidx) + ".csv")
                    layer_scores.append(float(csv["ap"]))
                except FileNotFoundError:
                    print(resultdir + "/" + layer + "_" + xai_method + "_" + str(classidx) + ".csv not found")
                    layer_scores.append(0.)

                # print(float(csv["separability_score"]))

            scores.append(np.mean(layer_scores))

        plt.plot(layers, scores)

    plt.xlabel("xai method")
    plt.ylabel("class-wise mean of separability scores")
    plt.title("ap")
    plt.legend(configs["xai_methods"])
    plt.show()

    plt.figure()

    for m, xai_method in enumerate(configs["xai_methods"]):
        scores = []

        for layer in layers:

            layer_scores = []

            for c, classidx in enumerate(classindices):

                try:
                    csv = pd.read_csv(resultdir + "/" + layer + "_" + xai_method + "_" + str(classidx) + ".csv")
                    layer_scores.append(float(csv["apr"]))
                except FileNotFoundError:
                    print(resultdir + "/" + layer + "_" + xai_method + "_" + str(classidx) + ".csv not found")
                    layer_scores.append(0.)

                # print(float(csv["separability_score"]))

            scores.append(np.mean(layer_scores))

        plt.plot(layers, scores)

    plt.xlabel("xai method")
    plt.ylabel("class-wise mean of separability scores")
    plt.title("ap with true on >= 0.1")
    plt.legend(configs["xai_methods"])
    plt.show()

    plt.figure()

    for m, xai_method in enumerate(configs["xai_methods"]):
        scores = []

        for layer in layers:

            layer_scores = []

            for c, classidx in enumerate(classindices):

                try:
                    csv = pd.read_csv(resultdir + "/" + layer + "_" + xai_method + "_" + str(classidx) + ".csv")
                    layer_scores.append(float(csv["f1"]))
                except FileNotFoundError:
                    print(resultdir + "/" + layer + "_" + xai_method + "_" + str(classidx) + ".csv not found")
                    layer_scores.append(0.)

                # print(float(csv["separability_score"]))

            scores.append(np.mean(layer_scores))

        plt.plot(layers, scores)

    plt.xlabel("xai method")
    plt.ylabel("class-wise mean of separability scores")
    plt.title("f1")
    plt.legend(configs["xai_methods"])
    plt.show()

    plt.figure()

    for m, xai_method in enumerate(configs["xai_methods"]):
        scores = []

        for layer in layers:

            layer_scores = []

            for c, classidx in enumerate(classindices):

                try:
                    csv = pd.read_csv(resultdir + "/" + layer + "_" + xai_method + "_" + str(classidx) + ".csv")
                    layer_scores.append(float(csv["f1r"]))
                except FileNotFoundError:
                    print(resultdir + "/" + layer + "_" + xai_method + "_" + str(classidx) + ".csv not found")
                    layer_scores.append(0.)

                # print(float(csv["separability_score"]))

            scores.append(np.mean(layer_scores))

        plt.plot(layers, scores)

    plt.xlabel("xai method")
    plt.ylabel("class-wise mean of separability scores")
    plt.title("f1 with true at >= 0.1")
    plt.legend(configs["xai_methods"])
    plt.show()
