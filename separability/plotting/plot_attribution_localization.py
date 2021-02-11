import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from ..xaitestframework.helpers.universal_helper import join_path


filepath = "config_pointing_game.yaml"

with open(filepath) as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)

    resultdir = "../results/attribution_localization"

    resultdir = resultdir + "/" + configs["data"]["dataname"] + "_" + configs["model"]["modelname"]

    classindices = np.arange(20)

    xai_methods = configs["xai_methods"]

    total_scores = []
    weighted_scores = []

    for i, xai_method in enumerate(xai_methods):
        print(xai_method)

        csv = pd.read_csv(resultdir + "_" + xai_method + ".csv")
        # print(float(csv["separability_score"]))
        scores = csv["total_score"]
        weighted = csv["weighted score"]
        total_scores.append(scores[0])
        weighted_scores.append(weighted[0])

    plt.figure()
    plt.bar(xai_methods, total_scores, 0.8)

    plt.xticks(rotation="45", ha="right")
    plt.xlabel("xai method")
    plt.ylabel("Scores")
    plt.title("Attribution localization scores [total score]")
    plt.show()

    plt.figure()
    plt.bar(xai_methods, weighted_scores, 0.8)

    plt.xticks(rotation="45", ha="right")
    plt.xlabel("xai method")
    plt.ylabel("Scores")
    plt.title("Attribution localization scores [weighted score]")
    plt.show()
