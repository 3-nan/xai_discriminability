import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# filepath = "configs/config_experiments.yaml"
#
# with open(filepath) as file:
#     configs = yaml.load(file, Loader=yaml.FullLoader)
#
#     distributions = ["uniform", "inpaint_ns"]   # , "gaussian", "inpaint_telea", "inpaint_ns"]
#     cmap = plt.cm.get_cmap("Set1", len(distributions))
#
#     layers = configs["layers"]
#     print(layers)
#     # layer = "conv13"
#
#     for layer in layers:
#
#         resultdir = "../results/activation_eval/third_run"    # /{}".format(layer)
#
#         resultdir = resultdir + "/" + configs["data"]["dataname"] + "_" + configs["model"]["modelname"] + "_" + layer
#
#         classindices = range(20)
#
#         for xai_method in configs["xai_methods"]:
#
#             fig, ax1 = plt.subplots()
#             ax2 = ax1.twinx()
#             print(xai_method)
#
#             for d, distribution in enumerate(distributions):
#                 scores = []
#                 diffs = []
#                 zero_diffs = []
#
#                 for classidx in classindices:
#
#                     csv = pd.read_csv(resultdir + "_" + xai_method + "_" + distribution + "_" + str(classidx) + ".csv")
#                     # print(float(csv["separability_score"]))
#                     # print(csv)
#                     percentages = csv["flip_percentage"][4::5]
#                     thresholds = csv["threshold"]
#                     scores.append(csv["zero_ratio"][4::5])
#                     diffs.append(csv["diff"][4::5])
#                     zero_diffs.append(csv["zero_diff"][4::5])
#
#                 ax1.plot(percentages, np.mean(scores, axis=0), linestyle="dotted", color=cmap(d), label=distribution)
#                 ax1.fill_between(percentages, np.min(scores, axis=0), np.max(scores, axis=0), color=cmap(d), alpha=.5)
#                 # locs = ax1.get_xticks()
#                 scores = np.array(scores)
#                 print(scores.shape)
#                 print(scores.T.shape)
#                 # ax1.boxplot(scores, positions=percentages, widths=.01)
#
#                 ax2.plot(percentages, np.mean(diffs, axis=0), linestyle="solid", color=cmap(d), label=distribution)
#                 ax2.plot(percentages, np.mean(zero_diffs, axis=0), linestyle="dashed", color=cmap(d), label=distribution)
#                 ax2.fill_between(percentages, np.min(zero_diffs, axis=0), np.max(zero_diffs, axis=0), color=cmap(d), alpha=.5)
#
#             # print(locs)
#             # ax1.set_xticks(locs)
#             # ax2.set_xticks(locs)
#             plt.xlabel("percentage of flipped pixels")
#             ax1.set_ylabel("kappa/gamma/delta score")
#             ax1.set_ylim(0.4, 1.)
#             # ax1.set_xlim(-0.05, 1.)
#
#             handles, labels = ax1.get_legend_handles_labels()
#             print(handles)
#             legend1 = plt.legend(handles, distributions, loc=2)
#
#             # line1 = Line2D([0, 1], [0, 1], linestyle="dotted", color="black")
#             # line2 = Line2D([0, 1], [0, 1], linestyle="dashdot", color="black")
#             # line3 = Line2D([0, 1], [0, 1], linestyle="dashed", color="black")
#             #
#             # legend2 = plt.legend([line1, line2, line3], ["kappa", "delta", "gamma"], loc=1)
#             # fig.gca().add_artist(legend1)
#             plt.title("Nearest Neighbour Scores: {}\n Layer: {}".format(xai_method, layer))
#
#             plt.show()

filepath = "configs/config_experiments.yaml"

with open(filepath) as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)

    distributions = ["uniform", "inpaint_ns"]   # , "gaussian", "inpaint_telea", "inpaint_ns"]
    cmap = plt.cm.get_cmap("Set1", len(distributions))

    layers = configs["layers"]
    print(layers)
    # layer = "conv13"

    for layer in layers:

        resultdir = "../results/activation_eval/Saliency"   # fourth_run"    # /{}".format(layer)

        resultdir = resultdir # + "/" + configs["data"]["dataname"] + "_" + configs["model"]["modelname"] + "_" + layer

        classindices = range(20)

        for xai_method in configs["xai_methods"]:

            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            print(xai_method)

            for d, distribution in enumerate(distributions):
                scores = {}
                diffs = {}
                zero_diffs = {}

                for classidx in classindices:

                    csv = pd.read_csv(os.path.join(resultdir, "{}_{}_{}_{}_{}_{}.csv".format(
                        configs["data"]["dataname"],
                        configs["model"]["modelname"],
                        layer,
                        xai_method,
                        distribution,
                        classidx
                    )))
                    # resultdir + "_" + xai_method + "_" + distribution + "_" + str(classidx) + ".csv")
                    # print(float(csv["separability_score"]))
                    # print(csv)
                    percentages = csv["flip_percentage"]
                    # thresholds = csv["threshold"]
                    # scores.append(csv["zero_ratio"][4::5])
                    # diffs.append(csv["diff"][4::5])
                    # zero_diffs.append(csv["zero_diff"][4::5])

                    for percentage in percentages:

                        if percentage not in list(scores.keys()):
                            scores[percentage] = []
                            diffs[percentage] = []
                            zero_diffs[percentage] = []

                        # b = np.load("{}/{}_{}_{}_{}_{}_{}_{}_{}.npy".format(
                        #     resultdir,
                        #     "scores",
                        #      configs["data"]["dataname"],
                        #      configs["model"]["modelname"],
                        #      layer,
                        #      xai_method,
                        #      distribution,
                        #      classidx,
                        #      percentage), allow_pickle=True).item()[percentage]
                        #
                        # print(list(scores.keys()))
                        # print(np.array(b).shape)
                        # raise ValueError()

                        scores[percentage].append(np.load("{}/{}_{}_{}_{}_{}_{}_{}_{}.npy".format(
                            resultdir,
                            "scores",
                             configs["data"]["dataname"],
                             configs["model"]["modelname"],
                             layer,
                             xai_method,
                             distribution,
                             classidx,
                             percentage), allow_pickle=True).item()[percentage])

                        diffs[percentage].append(np.load("{}/{}_{}_{}_{}_{}_{}_{}_{}.npy".format(
                            resultdir,
                            "diffs",
                             configs["data"]["dataname"],
                             configs["model"]["modelname"],
                             layer,
                             xai_method,
                             distribution,
                             classidx,
                             percentage), allow_pickle=True).item()[percentage])

                        zero_diffs[percentage].append(np.load("{}/{}_{}_{}_{}_{}_{}_{}_{}.npy".format(
                            resultdir,
                            "zero_diffs",
                            configs["data"]["dataname"],
                            configs["model"]["modelname"],
                            layer,
                            xai_method,
                            distribution,
                            classidx,
                            percentage), allow_pickle=True).item()[percentage])

                for percentage in percentages:
                    # print([np.array(a).shape for a in scores[percentage]])
                    scores[percentage] = np.concatenate(scores[percentage])
                    diffs[percentage] = np.concatenate(diffs[percentage])
                    zero_diffs[percentage] = np.concatenate(zero_diffs[percentage])
                    # print(scores[percentage].shape)

                percentages = percentages.to_numpy()
                print(type(percentages))

                ax1.plot(percentages, [np.median(scores[p]) for p in percentages], linestyle="dotted", color=cmap(d), label=distribution)
                ax1.fill_between(percentages, [np.percentile(scores[p], 25) for p in percentages], [np.percentile(scores[pe], 75) for pe in percentages], color=cmap(d), alpha=.5)
                # locs = ax1.get_xticks()
                # scores = np.array(scores)
                # print(scores.shape)
                # print(scores.T.shape)
                # ax1.boxplot(scores, positions=percentages, widths=.01)

                ax2.plot(percentages, [np.mean(diffs[percentage]) for percentage in percentages], linestyle="solid", color=cmap(d), label=distribution)
                ax2.fill_between(percentages, [np.percentile(diffs[p], 25) for p in percentages],
                                 [np.percentile(diffs[p], 75) for p in percentages], color=cmap(d), alpha=.5)
                ax2.plot(percentages, [np.mean(zero_diffs[percentage]) for percentage in percentages], linestyle="dashed", color=cmap(d), label=distribution)
                ax2.fill_between(percentages, [np.percentile(zero_diffs[p], 25) for p in percentages], [np.percentile(zero_diffs[p], 75) for p in percentages], color=cmap(d), alpha=.5)

            # print(locs)
            # ax1.set_xticks(locs)
            # ax2.set_xticks(locs)
            ax1.set_xlabel("Ratio of flipped pixels")
            ax1.set_ylabel("Ratio of zero activations")
            ax1.set_ylim(0.4, 1.)
            # ax1.set_xlim(-0.05, 1.)

            ax2.set_ylabel("Avg distance")

            handles, labels = ax1.get_legend_handles_labels()
            print(handles)
            legend1 = plt.legend(handles, distributions, loc=2)

            line1 = Line2D([0, 1], [0, 1], linestyle="dotted", color="black")
            line2 = Line2D([0, 1], [0, 1], linestyle="solid", color="black")
            line3 = Line2D([0, 1], [0, 1], linestyle="dashed", color="black")
            #
            legend2 = plt.legend([line1, line2, line3], ["% zero activations", "distance to original", "distance to 0"], loc=1)
            fig.gca().add_artist(legend1)
            # plt.title("Nearest Neighbour Scores: {}\n Layer: {}".format(xai_method, layer))

            plt.show()
            # plt.savefig("../results/figures/activation_analysis/{}.svg".format(layer))
