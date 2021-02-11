import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from ..xaitestframework.helpers.universal_helper import join_path


filepath = "eval.yaml"

with open(filepath) as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)

    resultdir = "../results/model_parameter_randomization/cascading_bottom_up"

    resultdir = resultdir + "/" + configs["data"]["dataname"] + "_" + configs["model"]["modelname"]

    # layers = [configs["layers"][0]]
    classindices = range(20)

    plt.figure()

    for xai_method in configs["xai_methods"]:

        method_scores = []

        for classidx in classindices:

            try:
                csv = pd.read_csv(resultdir + "_" + xai_method + "_" + str(classidx) + ".csv")

                method_scores.append(csv["score"])

            except FileNotFoundError:
                print("file for " + xai_method + " with idx " + str(classidx) + " not found")

        method_scores = np.array(method_scores)
        method_scores = np.mean(method_scores, axis=0)

        plt.plot(csv["layer"], method_scores)

    plt.legend(configs["xai_methods"])
    plt.show()

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
