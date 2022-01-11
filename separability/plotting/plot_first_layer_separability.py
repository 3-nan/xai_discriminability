import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from ..xaitestframework.helpers.universal_helper import join_path


filepath = "configs/config_experiments.yaml"

dataname = "VOC2012"
# modelname = "vgg16"
# dataname = "imagenet"
# modelname = "vgg16bn"
# modelname = "vgg16bn_uncanonized"
modelname = "vgg16"

with_ref = False

with open(filepath) as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)

    resultdir = "../results/separability"

    resultdir = resultdir + "/" + dataname + "_" + modelname

    if with_ref:
        ref_resultdir = resultdir + "_{}".format("uncanonized")

    layer = "conv1"

    if dataname == "VOC2012":
        classindices = range(20)
    elif dataname == "imagenet":
        classindices = [96, 126, 155, 292, 301, 347, 387, 405, 417, 426,
                        446, 546, 565, 573, 604, 758, 844, 890, 937, 954]
    else:
        print("please specify classindices for dataname {}!".format(dataname))

    # x = np.arange(len(layers))
    xai_methods = configs["xai_methods"]
    n_classes = len(classindices)

    width = 0.8
    cmap = plt.cm.get_cmap("Paired", 8)

    variants = ["ap", "f1"]

    for variant in variants:

        print("###############################################")
        print("#    Variant {}     #".format(variant))
        print("###############################################")

        plt.figure()

        for i, var in enumerate([variant, variant + "r"]):

            print("{}: {}".format(i, var))

            method_scores = []

            for m, xai_method in enumerate(xai_methods):
                scores = []

                for c, classidx in enumerate(classindices):

                    try:
                        csv = pd.read_csv(os.path.join(resultdir, layer + "_" + xai_method + "_" + str(classidx) + ".csv"))
                        if with_ref:
                            ref_csv = pd.read_csv(os.path.join(ref_resultdir, "{}_{}_{}.csv".format(layer, xai_method, classidx)))

                            if np.isnan(float(csv[var])) or np.isnan(float(ref_csv[var])):
                                print("NaN value for method {} layer {} class {}".format(xai_method, layer, classidx))
                            else:
                                scores.append(float(csv[var]) - float(ref_csv[var]))
                        else:
                            score = float(csv[var])
                            if not np.isnan(score):
                                scores.append(float(csv[var]))
                            else:
                                print("Warning: NaN value for method {} layer {} class {}".format(xai_method, layer, classidx))
                    except FileNotFoundError:
                        print(resultdir + "/" + layer + "_" + xai_method + "_" + str(classidx) + ".csv not found")
                        scores.append(1.)

                method_scores.append(np.mean(scores))

            ind = np.arange(len(xai_methods))
            if i == 1:
                plt.bar(ind + 0.2, method_scores, width=0.4, color=cmap(0))
            else:
                plt.bar(ind - 0.2, method_scores, width=0.4, color=cmap(1))
        plt.xticks(ind, labels=xai_methods, rotation="25", ha="right")
        plt.xlabel("XAI method")
        plt.ylabel("class-wise mean of separability scores")
        plt.legend(["theta=0", "theta=0.1"])
        # plt.ylim([0.475, 1.025])
        if variant == "ap":
            plt.title("Average Precision")
        elif variant == "f1":
            plt.title("Separability with F1-Score")

        plt.tight_layout()
        # plt.show()
        if with_ref:
            plt.savefig("../results/separability/figures/{}_{}/input_ref_{}.svg".format(dataname, modelname, variant), format="svg")

        else:
            plt.savefig("../results/separability/figures/{}_{}/input_{}.svg".format(dataname, modelname, variant), format="svg")
        plt.close()
