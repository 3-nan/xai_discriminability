import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from ..xaitestframework.helpers.universal_helper import join_path

filepath = "configs/config_experiments.yaml"

dataname = "imagenet"
modelname = "resnet18"          # "vgg16bn"

data = "{}_{}".format(modelname, dataname)
ref_data = "{}_{}_uncanonized".format(modelname, dataname)

with open(filepath) as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)

    if data:
        resultdir = "../results/{}/attribution_localization/{}_{}".format(data, dataname, modelname)
        ref_resultdir = "../results/{}/attribution_localization/{}_{}".format(ref_data, dataname, modelname)

    if dataname == "VOC2012":
        classindices = range(20)
    elif dataname == "imagenet":
        classindices = [96, 126, 155, 292, 301, 347, 387, 405, 417, 426,
                        446, 546, 565, 573, 604, 758, 844, 890, 937, 954]
    else:
        print("please specify classindices for dataname {}!".format(dataname))

    xai_methods = configs["xai_methods"]

    total_scores = []
    weighted_scores = []
    scores_u50 = []
    scores_u25 = []

    ref_scores, ref_weighted, ref_u50, ref_u25 = [], [], [], []

    for i, xai_method in enumerate(xai_methods):
        print(xai_method)

        csv = pd.read_csv(resultdir + "_" + xai_method + ".csv")
        ref_csv = pd.read_csv("{}_{}.csv".format(ref_resultdir, xai_method))
        # print(float(csv["separability_score"]))
        scores = csv["total_score"]
        weighted = csv["weighted score"]

        total_scores.append(scores[0])
        weighted_scores.append(weighted[0])
        scores_u50.append(csv["score_u50"][0])
        scores_u25.append(csv["score_u25"][0])

        ref_scores.append(ref_csv["total_score"][0])
        ref_weighted.append(ref_csv["weighted score"][0])
        ref_u50.append(ref_csv["score_u50"][0])
        ref_u25.append(ref_csv["score_u25"][0])

    cmap = plt.cm.get_cmap("Paired", 12)

    fig, ax = plt.subplots()

    ind = np.arange(len(xai_methods))

    c1 = ax.bar(ind - 0.25, total_scores, 0.25, tick_label=xai_methods, label="canonized", color=cmap(0))
    u1 = ax.bar(ind - 0.1875, ref_scores, 0.125, tick_label=xai_methods, label="uncanonized", color=cmap(1))
    c2 = ax.bar(ind, scores_u50, 0.25, tick_label=xai_methods, label="canonized", color=cmap(2))
    u2 = ax.bar(ind + 0.0625, ref_u50, 0.125, tick_label=xai_methods, label="uncanonized", color=cmap(3))
    c3 = ax.bar(ind + 0.25, scores_u25, 0.25, tick_label=xai_methods, label="canonized", color=cmap(4))
    u3 = ax.bar(ind + 0.3125, ref_u25, 0.125, tick_label=xai_methods, label="uncanonized", color=cmap(5))

    # ax.bar(ind + 0.5, scores_u25, 0.25, tick_label=xai_methods, color=cmap(6))
    # ax.bar(ind + 0.5, ref_u25, 0.125, tick_label=xai_methods, color=cmap(7))

    plt.xticks(ind, labels=xai_methods, rotation="25", ha="right")
    plt.ylim([0.0, 1.0])
    plt.xlabel("XAI method")
    plt.ylabel("Scores")
    plt.title("Attribution Localization (class-wise mean)")
    plt.legend(handles=[c1, u1, c2, u2, c3, u3],
               labels=["", "", "", "", "canonized", "uncanonized"],
               ncol=3, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5)
    plt.tight_layout()
    plt.show()

    d = {"xai_method": xai_methods,
         "total_scores": total_scores,
         "weighted_scores": weighted_scores,
         "scores_u50": scores_u50,
         "scores_u25": scores_u25
         }
    df = pd.DataFrame(data=d)
    df.to_csv("../results/{}/attribution_localization_results.csv".format(data), index=False)
