import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from ..xaitestframework.helpers.universal_helper import join_path


filepath = "eval.yaml"

with open(filepath) as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)

    resultdir = "../results/model_parameter_randomization/independent"

    resultdir = resultdir + "/" + configs["data"]["dataname"] + "_" + configs["model"]["modelname"]

    # layers = [configs["layers"][0]]
    classindices = range(20)

    for classidx in classindices:

        plt.figure()

        for xai_method in configs["xai_methods"]:

            print(xai_method)
            # scores = []

            csv = pd.read_csv(resultdir + "_" + xai_method + "_" + str(classidx) + ".csv")

            # scores.append(float(csv["separability_score"]))

            plt.plot(csv["layer"], csv["score"])

        plt.xlabel("layer")
        plt.ylabel("score")
        plt.legend(configs["xai_methods"])
        plt.title("MPR mean difference (mse) of explanations for class {}".format(classidx))
        plt.show()
