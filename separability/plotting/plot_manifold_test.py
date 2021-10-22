import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# from ..xaitestframework.helpers.universal_helper import join_path

# sns.set_theme(style="darkgrid")
# sns.set_palette(sns.color_palette("hls", 20))

filepath = "configs/config_experiments.yaml"

layer = "linear2"

with open(filepath) as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)

    distributions = ["uniform", "gaussian", "inpaint_telea", "inpaint_ns"]   # , "gaussian", "inpaint_telea", "inpaint_ns"]
    cmap = plt.cm.get_cmap("rainbow", len(distributions))

    resultdir = "../results/manifold_outlier_pixelflipping_experiment/Saliency/{}".format(layer)

    resultdir = resultdir + "/" + configs["data"]["dataname"] + "_" + configs["model"]["modelname"]

    layers = [configs["layers"][0]]
    classindices = range(20)

    for xai_method in configs["xai_methods"]:

        fig, ax1 = plt.subplots()
        # ax2 = ax1.twinx()
        print(xai_method)

        for d, distribution in enumerate(distributions):
            scores = {"kappas": [],
                      "gammas": [],
                      "deltas": [],

                      "initial_kappas": [],
                      "initial_gammas": [],
                      "initial_deltas": []
                      }
            for classidx in classindices:

                csv = pd.read_csv(resultdir + "_" + xai_method + "_" + distribution + "_" + str(classidx) + ".csv")
                # print(float(csv["separability_score"]))
                # print(csv)
                percentages = csv["flip_percentage"]
                scores["kappas"].append(csv["kappa"])
                scores["gammas"].append(csv["gamma"])
                scores["deltas"].append(csv["delta"])

                scores["initial_kappas"].append(csv["initial_kappa"])
                scores["initial_gammas"].append(csv["initial_gamma"])
                scores["initial_deltas"].append(csv["initial_delta"])

            ax1.plot(percentages, np.mean(scores["kappas"], axis=0), linestyle="solid", color=cmap(d), label=distribution)
            ax1.plot(percentages, np.mean(scores["gammas"], axis=0), linestyle="dashed", color=cmap(d), label=distribution)
            ax1.plot(percentages, np.mean(scores["deltas"], axis=0), linestyle="dashdot", color=cmap(d), label=distribution)
            ax1.plot(percentages, np.mean(scores["initial_kappas"], axis=0), linestyle="solid", color=cmap(d),
                     label=distribution, alpha=.4)
            ax1.plot(percentages, np.mean(scores["initial_gammas"], axis=0), linestyle="dashed", color=cmap(d),
                     label=distribution, alpha=.4)
            ax1.plot(percentages, np.mean(scores["initial_deltas"], axis=0), linestyle="dashdot", color=cmap(d),
                     label=distribution, alpha=.4)


        plt.xlabel("Ratio of flipped pixels")
        ax1.set_ylabel(r"$\kappa$/$\gamma$/$\delta$ Score")

        handles, labels = ax1.get_legend_handles_labels()
        legend1 = plt.legend(handles[1::6], distributions, loc=2)

        line1 = Line2D([0, 1], [0, 1], linestyle="solid", color="black")
        line2 = Line2D([0, 1], [0, 1], linestyle="dashdot", color="black")
        line3 = Line2D([0, 1], [0, 1], linestyle="dashed", color="black")

        legend2 = plt.legend([line1, line2, line3], [r"$\kappa$", r"$\delta$", r"$\gamma$"], loc=1)
        fig.gca().add_artist(legend1)
        # plt.title("Nearest Neighbour Scores: {}".format(xai_method))

        plt.show()
        # plt.savefig("../results/figures/manifold_pixelflipping_{}_final.svg".format(layer))
