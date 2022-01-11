import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from ..xaitestframework.helpers.universal_helper import join_path


filepath = "configs/config_experiments.yaml"

dataname = "VOC2012"
modelname = "vgg16_cpu"
# dataname = "imagenet"
# modelname = "resnet18"
# modelname = "vgg16bn"
# modelname = "vgg16bn_uncanonized"
# modelname = "vgg16"

with_ref = False

with open(filepath) as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)

    resultdir = "../results/separability"

    resultdir = resultdir + "/" + dataname + "_" + modelname

    if with_ref:
        ref_resultdir = resultdir + "_{}".format("uncanonized")

    layers = configs["layers"]

    if dataname == "VOC2012":
        classindices = range(20)
    elif dataname == "imagenet":
        classindices = [96, 126, 155, 292, 301, 347, 387, 405, 417, 426,
                        446, 546, 565, 573, 604, 758, 844, 890, 937, 954]
    else:
        print("please specify classindices for dataname {}!".format(dataname))

    print(layers)

    # layer = layers[0]
    #
    # print(layer)

    method_scores = []

    x = np.arange(len(layers))
    n_classes = len(classindices)

    width = 0.8
    cmap = plt.cm.get_cmap("Set1", len(configs["xai_methods"]))

    # for m, xai_method in enumerate(configs["xai_methods"]):
    #     scores = []
    #
    #     for c, classidx in enumerate(classindices):
    #
    #         class_scores = []
    #
    #         for layer in layers:
    #
    #             try:
    #                 csv = pd.read_csv(resultdir + "/" + layer + "_" + xai_method + "_" + str(classidx) + ".csv")
    #                 class_scores.append(float(csv["separability_score"]))
    #             except FileNotFoundError:
    #                 print(resultdir + "/" + layer + "_" + xai_method + "_" + str(classidx) + ".csv not found")
    #                 class_scores.append(0.)
    #             # print(float(csv["separability_score"]))
    #
    #         # print(len(layers))
    #         # print(len(class_scores))
    #         ax.bar(x - width / 2. + c * (width/n_classes), class_scores, width / n_classes, label=classidx, color=cmap(m), alpha=0.5)
    #
    #     # plt.plot(layers, scores)
    #
    # handles, labels = ax.get_legend_handles_labels()
    #
    # plt.xticks(x, layers)
    # plt.xlabel("Layer")
    # plt.ylabel("class-wise mean of margins")
    # plt.legend(handles[::n_classes], configs["xai_methods"])
    # plt.show()

    variants = ["ap", "apr", "f1", "f1r"]

    for variant in variants:

        print("###############################################")
        print("#    Variant {}     #".format(variant))
        print("###############################################")
        plt.figure()

        for m, xai_method in enumerate(configs["xai_methods"]):
            scores = []

            for layer in layers:

                layer_scores = []

                for c, classidx in enumerate(classindices):

                    try:
                        csv = pd.read_csv(os.path.join(resultdir, layer + "_" + xai_method + "_" + str(classidx) + ".csv"))
                        if with_ref:
                            ref_csv = pd.read_csv(os.path.join(ref_resultdir, "{}_{}_{}.csv".format(layer, xai_method, classidx)))

                            if np.isnan(float(csv[variant])) or np.isnan(float(ref_csv[variant])):
                                print("NaN value for method {} layer {} class {}".format(xai_method, layer, classidx))
                            else:
                                layer_scores.append(float(csv[variant]) - float(ref_csv[variant]))
                        else:
                            score = float(csv[variant])
                            # if not np.isnan(score):
                            layer_scores.append(float(csv[variant]))
                            if np.isnan(score):
                                print("Warning: NaN value for method {} layer {} class {}".format(xai_method, layer, classidx))
                    except FileNotFoundError:
                        print(resultdir + "/" + layer + "_" + xai_method + "_" + str(classidx) + ".csv not found")
                        layer_scores.append(1.)

                    # print(float(csv["separability_score"]))

                scores.append(np.mean(layer_scores))

            plt.plot(layers, scores)

        plt.xlabel("XAI method")
        if with_ref:
            plt.ylabel("Diff. in separability score")
        else:
            plt.ylabel("class-wise mean of separability scores")
        plt.ylim([0.25, 1.025])       # voc vgg16
        # plt.ylim([-0.15, 0.15])         # ref imagenet vgg16bn
        # plt.ylim([-0.175, 0.25])        # ref imagenet resnet18
        if variant == "ap":
            plt.title("Average Precision")
        elif variant == "apr":
            plt.title("Average Precision with theta = 0.1")
        elif variant == "f1":
            plt.title("F1 Score")
        elif variant == "f1r":
            plt.title("F1 Score with theta = 0.1")
        plt.legend(configs["xai_methods"])
        # plt.show()
        if with_ref:
            plt.savefig("../results/separability/figures/{}_{}/ref_{}.svg".format(dataname, modelname, variant), format="svg")

        else:
            plt.savefig("../results/separability/figures/{}_{}/{}.svg".format(dataname, modelname, variant), format="svg")
        plt.close()

    # raise ValueError()
    # plt.figure()
    #
    # for m, xai_method in enumerate(configs["xai_methods"]):
    #     scores = []
    #
    #     for layer in layers:
    #
    #         layer_scores = []
    #
    #         for c, classidx in enumerate(classindices):
    #
    #             try:
    #                 csv = pd.read_csv(resultdir + "/" + layer + "_" + xai_method + "_" + str(classidx) + ".csv")
    #                 score = float(csv["apr"])
    #                 if not np.isnan(score):
    #                     layer_scores.append(score)
    #                 else:
    #                     print("Warning: NaN value for method {} layer {} class {}".format(xai_method, layer, classidx))
    #
    #             except FileNotFoundError:
    #                 print(resultdir + "/" + layer + "_" + xai_method + "_" + str(classidx) + ".csv not found")
    #                 layer_scores.append(1.)
    #
    #         scores.append(np.mean(layer_scores))
    #
    #     plt.plot(layers, scores)
    #
    # plt.xlabel("XAI method")
    # plt.ylabel("class-wise mean of separability scores")
    # plt.ylim([0.375, 1.025])
    # plt.title("Average Precision with theta = 0.1")
    # plt.legend(configs["xai_methods"])
    # # plt.show()
    # plt.savefig("../results/separability/figures/{}_{}/ap_filtered_all.svg".format(dataname, modelname), format="svg")
    # plt.close()
    #
    # plt.figure()
    #
    # for m, xai_method in enumerate(configs["xai_methods"]):
    #     scores = []
    #
    #     for layer in layers:
    #
    #         layer_scores = []
    #
    #         for c, classidx in enumerate(classindices):
    #
    #             try:
    #                 csv = pd.read_csv(resultdir + "/" + layer + "_" + xai_method + "_" + str(classidx) + ".csv")
    #                 layer_scores.append(float(csv["f1"]))
    #             except FileNotFoundError:
    #                 print(resultdir + "/" + layer + "_" + xai_method + "_" + str(classidx) + ".csv not found")
    #                 layer_scores.append(1.)
    #
    #             # print(float(csv["separability_score"]))
    #
    #         scores.append(np.mean(layer_scores))
    #
    #     plt.plot(layers, scores)
    #
    # plt.xlabel("XAI method")
    # plt.ylabel("class-wise mean of separability scores")
    # plt.ylim([-0.025, 1.025])
    # plt.title("F1 Score")
    # plt.legend(configs["xai_methods"])
    # # plt.show()
    # plt.savefig("../results/separability/figures/{}_{}/f1_all.svg".format(dataname, modelname), format="svg")
    # plt.close()
    #
    # plt.figure()
    #
    # for m, xai_method in enumerate(configs["xai_methods"]):
    #     scores = []
    #
    #     for layer in layers:
    #
    #         layer_scores = []
    #
    #         for c, classidx in enumerate(classindices):
    #
    #             try:
    #                 csv = pd.read_csv(resultdir + "/" + layer + "_" + xai_method + "_" + str(classidx) + ".csv")
    #                 layer_scores.append(float(csv["f1r"]))
    #             except FileNotFoundError:
    #                 print(resultdir + "/" + layer + "_" + xai_method + "_" + str(classidx) + ".csv not found")
    #                 layer_scores.append(1.)
    #
    #             # print(float(csv["separability_score"]))
    #
    #         scores.append(np.mean(layer_scores))
    #
    #     plt.plot(layers, scores)
    #
    # plt.xlabel("XAI method")
    # plt.ylabel("class-wise mean of separability scores")
    # plt.ylim([-0.025, 1.025])
    # plt.title("F1 Score with theta = 0.1")
    # plt.legend(configs["xai_methods"])
    # # plt.show()
    # plt.savefig("../results/separability/figures/{}_{}/f1_filtered_all.svg".format(dataname, modelname), format="svg")
    # plt.close()
