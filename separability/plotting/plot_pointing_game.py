import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from ..xaitestframework.helpers.universal_helper import join_path


filepath = "configs/config_experiments.yaml"

with open(filepath) as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)

    resultdir = "../results/pointing_game/with_blur"

    resultdir = resultdir + "/" + configs["data"]["dataname"] + "_" + configs["model"]["modelname"]

    classindices = np.arange(20)

    xai_methods = configs["xai_methods"]

    width = 0.8 / len(xai_methods)

    mean_scores = []
    mean_scores_u50 = []
    mean_scores_u25 = []

    # fig, ax = plt.subplots()

    for i, xai_method in enumerate(xai_methods):
        print(xai_method)

        csv = pd.read_csv(resultdir + "_" + xai_method + ".csv")
        # print(float(csv["separability_score"]))
        scores = csv["score"]

        mean_scores.append(np.mean(scores))

        mean_scores_u50.append(np.mean(csv["score_u50"]))
        mean_scores_u25.append(np.mean(csv["score_u25"]))

        # rects = ax.bar(classindices - 0.4 + i*width + width/2, scores, width, label=xai_method)

    # ax.set_xlabel("Class index")
    # ax.set_ylabel("Scores")
    # ax.set_title("Pointing Game Scores")
    # ax.set_xticks(classindices)
    # ax.legend()
    #
    # plt.show()
    #
    # plt.figure()
    # plt.bar(xai_methods, mean_scores)
    #
    # plt.xticks(rotation="45", ha="right")
    # plt.xlabel("xai method")
    # plt.ylabel("Scores")
    # plt.title("Pointing Game (class-wise mean)")
    # plt.show()

    # write csv
    d = {"xai_method": xai_methods,
         "scores": mean_scores,
         "scores_u50": mean_scores_u50,
         "scores_u25": mean_scores_u25
         }
    df = pd.DataFrame(data=d)

    df.to_csv("../results/pointing_game_with_blur.csv", index=False)
