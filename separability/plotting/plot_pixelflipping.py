import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from ..xaitestframework.helpers.universal_helper import join_path


filepath = "config.yaml"

with open(filepath) as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)

    resultdir = "../results/pixelflipping"

    resultdir = resultdir + "/" + configs["data"]["dataname"] + "_" + configs["model"]["modelname"]

    layers = [configs["layers"][0]]
    classindices = range(20)

    for xai_method in configs["xai_methods"]:

        plt.figure()
        print(xai_method)
        scores = []
        for classidx in classindices:
            csv = pd.read_csv(resultdir + "_" + xai_method + "_" + "uniform" + str(classidx) + ".csv")
            # print(float(csv["separability_score"]))
            # print(csv)
            percentages = csv["flip_percentage"]
            scores = csv["flipped_score"]

            plt.plot(percentages, scores)

        plt.xlabel("percentage of flipped pixels")
        plt.ylabel("score")
        plt.title(xai_method)

        plt.show()

    # build mean over classes
    plt.figure()

    for xai_method in configs["xai_methods"]:

        method_scores = []

        for classidx in classindices:
            csv = pd.read_csv(resultdir + "_" + xai_method + "_" + "uniform" + str(classidx) + ".csv")
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
    plt.title("Pixelflipping (class-wise mean)")
    plt.legend(configs["xai_methods"])

    plt.show()
