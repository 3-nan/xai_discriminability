import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from ..xaitestframework.helpers.universal_helper import join_path


filepath = "configs/config_experiments.yaml"

data = "resnet18_imagenet"  #  "vgg16bn_imagenet"

ref_data = "resnet18_imagenet_uncanonized"      # "vgg16bn_imagenet_uncanonized"

dataname = "imagenet"
modelname = "resnet18"            # "vgg16bn"

blur = False

with open(filepath) as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)

    cmap = plt.cm.get_cmap("Paired", 12)

    if data:
        print("on {}".format(data))
        resultdir = "../results/{}/pointing_game".format(data)
        ref_resultdir = "../results/{}/pointing_game".format(ref_data)
    else:
        resultdir = "../results/pointing_game"

    # if blur:
    #     resultdir += "/blur"

    print(resultdir)

    resultdir = resultdir + "/{}_{}".format(dataname, modelname)
    ref_resultdir = ref_resultdir + "/{}_{}".format(dataname, modelname)

    xai_methods = configs["xai_methods"]

    width = 0.8 / len(xai_methods)

    mean_scores = []
    mean_scores_u50 = []
    mean_scores_u25 = []

    ref_scores, ref_scores_u50, ref_scores_u25 = [], [], []

    # fig, ax = plt.subplots()

    for i, xai_method in enumerate(xai_methods):

        if blur:
            print("{} with blur".format(xai_method))
            csv = pd.read_csv(resultdir + "_" + xai_method + "_blur.csv")
        else:
            print("{} without blur".format(xai_method))
            csv = pd.read_csv(resultdir + "_" + xai_method + ".csv")
            ref_csv = pd.read_csv("{}_{}.csv".format(ref_resultdir, xai_method))

        # print(float(csv["separability_score"]))
        scores = csv["score"]

        mean_scores.append(np.mean(scores))

        mean_scores_u50.append(np.mean(csv["score_u50"]))
        mean_scores_u25.append(np.mean(csv["score_u25"]))

        ref_scores.append(np.mean(ref_csv["score"]))
        ref_scores_u50.append(np.mean(ref_csv["score_u50"]))
        ref_scores_u25.append(np.mean(ref_csv["score_u25"]))
        # rects = ax.bar(classindices - 0.4 + i*width + width/2, scores, width, label=xai_method)

    # ax.set_xlabel("Class index")
    # ax.set_ylabel("Scores")
    # ax.set_title("Pointing Game Scores")
    # ax.set_xticks(classindices)
    # ax.legend()
    #
    # plt.show()
    #
    ind = np.arange(len(xai_methods))

    print(mean_scores)

    fig, ax = plt.subplots()

    # for x, xai_method in enumerate(xai_methods):
    c1 = ax.bar(ind - 0.25, mean_scores, 0.25, tick_label=xai_methods, label="canonized", color=cmap(0))
    u1 = ax.bar(ind - 0.1875, ref_scores, 0.125, tick_label=xai_methods, label="uncanonized", color=cmap(1))
    c2 = ax.bar(ind, mean_scores_u50, 0.25, tick_label=xai_methods, label="canonized", color=cmap(2))
    u2 = ax.bar(ind + 0.0625, ref_scores_u50, 0.125, tick_label=xai_methods, label="uncanonized", color=cmap(3))
    c3 = ax.bar(ind + 0.25, mean_scores_u25, 0.25, tick_label=xai_methods, label="canonized", color=cmap(4))
    u3 = ax.bar(ind + 0.3125, ref_scores_u25, 0.125, tick_label=xai_methods, label="uncanonized", color=cmap(5))

    plt.xticks(ind, labels=xai_methods, rotation="25", ha="right")
    plt.ylim([0.0, 1.0])
    plt.xlabel("XAI method")
    plt.ylabel("Scores")
    plt.title("Pointing Game (class-wise mean)")
    plt.legend(handles=[c1, u1, c2, u2, c3, u3],
               labels=["", "", "", "", "canonized", "uncanonized"],
               ncol=3, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5)
    plt.tight_layout()
    plt.show()

    # write csv
    d = {"xai_method": xai_methods,
         "scores": mean_scores,
         "scores_u50": mean_scores_u50,
         "scores_u25": mean_scores_u25
         }
    df = pd.DataFrame(data=d)

    if data:
        if blur:
            df.to_csv("../results/{}/pointing_game/pointing_game_with_blur.csv".format(data), index=False)
        else:
            df.to_csv("../results/{}/pointing_game/pointing_game.csv".format(data), index=False)
    else:
        if blur:
            df.to_csv("../results/pointing_game/pointing_game_with_blur.csv", index=False)
        else:
            df.to_csv("../results/pointing_game/pointing_game.csv", index=False)
