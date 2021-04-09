import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from ..xaitestframework.helpers.universal_helper import join_path

# sns.set_theme(style="darkgrid")
# sns.set_palette(sns.color_palette("hls", 20))

filepath = "configs/config_experiments.yaml"

with open(filepath) as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)

    distribution = "inpaint_ns"
    resultdir = "../results/pixelflipping"

    resultdir = resultdir + "/" + configs["data"]["dataname"] + "_" + configs["model"]["modelname"]

    log_scale = False

    layers = [configs["layers"][0]]
    classindices = range(20)

    # for xai_method in configs["xai_methods"]:
    #
    #     plt.figure()
    #     print(xai_method)
    #     scores = []
    #     for classidx in classindices:
    #         try:
    #             csv = pd.read_csv(resultdir + "_" + xai_method + "_" + distribution + str(classidx) + ".csv")
    #         except FileNotFoundError:
    #             csv = pd.read_csv(resultdir + "_" + xai_method + "_" + distribution + "_" + str(classidx) + ".csv")
    #         # print(float(csv["separability_score"]))
    #         # print(csv)
    #         percentages = csv["flip_percentage"]
    #         scores = csv["flipped_score"]
    #
    #         plt.plot(percentages, scores)
    #
    #     plt.xlabel("percentage of flipped pixels")
    #     plt.ylabel("score")
    #     plt.title(xai_method + " with {} sampling".format(distribution))
    #
    #     plt.show()

    # build mean over classes
    plt.figure()

    for xai_method in configs["xai_methods"]:

        method_scores = []

        for classidx in classindices:
            try:
                csv = pd.read_csv(resultdir + "_" + xai_method + "_" + distribution + str(classidx) + ".csv")
            except FileNotFoundError:
                csv = pd.read_csv(resultdir + "_" + xai_method + "_" + distribution + "_" + str(classidx) + ".csv")
            # print(float(csv["separability_score"]))
            # print(csv)
            percentages = csv["flip_percentage"]
            scores = csv["flipped_score"]

            method_scores.append(scores)

        method_scores = np.array(method_scores)
        method_scores = np.mean(method_scores, axis=0)

        plt.plot(percentages, method_scores)

    plt.xlabel("percentage of flipped pixels")
    plt.ylabel("score")

    if log_scale:
        plt.xscale("log")

    plt.title("Pixelflipping (class-wise mean) with {} sampling".format(distribution))
    plt.legend(configs["xai_methods"])

    # plt.show()

    plt.savefig("../results/figures/pixelflipping_{}".format(distribution), format="svg")


def lineplot(data, xlabel, ylabel):

    plot = sns.lineplot(x=xlabel, y=ylabel, data=data)

    return plot
