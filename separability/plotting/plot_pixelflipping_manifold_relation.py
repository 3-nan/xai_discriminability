import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# from ..xaitestframework.helpers.universal_helper import join_path

# sns.set_theme(style="darkgrid")
# sns.set_palette(sns.color_palette("hls", 20))

filepath = "configs/config_experiments.yaml"

with open(filepath) as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)

    distributions = ["uniform", "gaussian", "inpaint_telea", "inpaint_ns"]
    cmap = plt.cm.get_cmap("Set1", len(distributions))

    pixelflipping_resultdir = "../results/pixelflipping"
    pixelflipping_resultdir = pixelflipping_resultdir + "/" + configs["data"]["dataname"] + "_" + configs["model"]["modelname"]

    manifold_resultdir = "../results/manifold_outlier_pixelflipping_experiment"
    manifold_resultdir = manifold_resultdir + "/" + configs["data"]["dataname"] + "_" + configs["model"]["modelname"]

    layers = [configs["layers"][0]]
    classindices = range(20)

    for xai_method in configs["xai_methods"]:

        fig, ax1 = plt.subplots()
        # ax2 = ax1.twinx()
        print(xai_method)

        for d, distribution in enumerate(distributions):
            scores = {"flipped_scores": [],
                      "kappas": [],
                      "gammas": [],
                      "deltas": []
                      }
            for classidx in classindices:

                pixelflipping_csv = pd.read_csv(pixelflipping_resultdir + "_" + xai_method + "_" + distribution + "_" + str(classidx) + ".csv")
                manifold_csv = pd.read_csv(manifold_resultdir + "_" + xai_method + "_" + distribution + "_" + str(classidx) + ".csv")
                # print(float(csv["separability_score"]))
                # print(csv)
                percentages_p = pixelflipping_csv["flip_percentage"]
                percentages_m = pixelflipping_csv["flip_percentage"]
                assert (percentages_p == percentages_m).all()

                scores["flipped_scores"].append(pixelflipping_csv["flipped_score"])

                scores["kappas"].append(manifold_csv["kappa"])
                # scores["gammas"].append(manifold_csv["gamma"])
                # scores["deltas"].append(manifold_csv["delta"])

            ax1.plot(np.mean(scores["flipped_scores"], axis=0), np.mean(scores["kappas"], axis=0), linestyle="dotted", color=cmap(d), label=distribution, marker="x")
            # ax1.plot(np.mean(scores["flipped_scores"], axis=0), np.mean(scores["gammas"], axis=0), linestyle="dashed", color=cmap(d), label=distribution)
            # ax1.plot(np.mean(scores["flipped_scores"], axis=0), np.mean(scores["deltas"], axis=0), linestyle="dashdot", color=cmap(d), label=distribution)

        plt.xlabel("flipped score")
        ax1.set_ylabel("kappa/gamma/delta score")

        # handles, labels = ax1.get_legend_handles_labels()
        # legend1 = plt.legend(handles[1::3], distributions, loc=2)
        plt.legend(distributions)

        # line1 = Line2D([0, 1], [0, 1], linestyle="dotted", color="black")
        # line2 = Line2D([0, 1], [0, 1], linestyle="dashdot", color="black")
        # line3 = Line2D([0, 1], [0, 1], linestyle="dashed", color="black")
        #
        # legend2 = plt.legend([line1, line2, line3], ["kappa", "delta", "gamma"], loc=1)
        # fig.gca().add_artist(legend1)
        plt.title("Pixelflipping Data Manifold Relation: {}".format(xai_method))

        # ax1.set_yscale("log")
        ax1.set_xlim(0.7, 0)

        plt.show()
