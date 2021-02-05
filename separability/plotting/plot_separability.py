import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from ..xaitestframework.helpers.universal_helper import join_path


filepath = "config.yaml"

with open(filepath) as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)

    resultdir = "../results/one_class_separability"

    resultdir = resultdir + "/" + configs["data"]["dataname"] + "_" + configs["model"]["modelname"]

    layers = [configs["layers"][0]]
    classindices = range(20)

    for layer in layers:

        plt.figure()

        for xai_method in configs["xai_methods"]:
            print(xai_method)
            scores = []
            for classidx in classindices:
                csv = pd.read_csv(resultdir + "/" + layer + "_" + xai_method + "_" + str(classidx) + ".csv")
                # print(float(csv["separability_score"]))
                scores.append(float(csv["separability_score"]))

            plt.plot(scores)

        plt.show()
