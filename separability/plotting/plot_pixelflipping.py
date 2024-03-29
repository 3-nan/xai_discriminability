import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from ..xaitestframework.helpers.universal_helper import join_path

# sns.set_theme(style="darkgrid")
# sns.set_palette(sns.color_palette("hls", 20))

# plt.rcParams.update({'font.size': 13})

LABEL_DICT = {
    "epsilon": r'LRP-$\varepsilon$',
    "epsilon_plus": r'LRP-$\varepsilon$-+',
    "epsilon_plus_flat": r'LRP-$\varepsilon$-+-$\flat$',
    "epsilon_alpha2_beta1": r'LRP-$\varepsilon$-$\alpha$2$\beta$1',
    "epsilon_alpha2_beta1_flat": r'LRP-$\varepsilon$-$\alpha$2$\beta$1-$\flat$',
    "guided_backprop": "GBP",
    "excitation_backprop": "EBP",
    "Gradient": "Gradient",
    "IntegratedGradients": "IG",
}

filepath = "configs/config_experiments.yaml"

dataname = "imagenet"       # VOC2012   imagenet
modelname = "resnet18"      # "vgg16bn"     vgg16   resnet18

# configuration = "imagenet_resnet18"   # "vgg16bn_imagenet"    resnet18_imagenet
# ref_configuration = "imagenet_resnet18_noncanonized"         # "vgg16bn_imagenet_uncanonized"
configuration = "imagenet_{}".format(modelname)
ref_configuration = "imagenet_{}_noncanonized".format(modelname)

distribution = "uniform"     # "uniform" "inpaint_ns"

with open(filepath) as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)

    if configuration:
        resultdir = "../results/canonization/{}/pixelflipping/{}_{}".format(configuration, dataname, modelname)
        ref_resultdir = "../results/canonization/{}/pixelflipping/{}_{}".format(ref_configuration, dataname, modelname)
    else:
        resultdir = "../results/pixelflipping/{}_{}".format(dataname, modelname)

    log_scale = False    # False

    layers = [configs["layers"][0]]

    if dataname == "VOC2012":
        classindices = range(20)
    elif dataname == "imagenet":
        classindices = [96, 126, 155, 292, 301, 347, 387, 405, 417, 426,
                        446, 546, 565, 573, 604, 758, 844, 890, 937, 954]
    else:
        print("please specify classindices for dataname {}!".format(dataname))

    # for xai_method in configs["xai_methods"]:
    #
    #     plt.figure()
    #     print(xai_method)
    #     scores = []
    #     for classidx in classindices:
    #         try:
    #             csv = pd.read_csv(resultdir + "_" + xai_method + "_" + distribution + str(classidx) + ".csv")
    #         except FileNotFoundError:
    #             csv = pd.read_csv(resultdir + "_" + xai_method + "_" + distribution + "_" + str(classidx) + ".csv")
    #         # print(float(csv["separability_score"]))
    #         # print(csv)
    #         percentages = csv["flip_percentage"]
    #         scores = csv["flipped_score"]
    #
    #         plt.plot(percentages, scores)
    #
    #     plt.xlabel("percentage of flipped pixels")
    #     plt.ylabel("score")
    #     plt.title(xai_method + " with {} sampling".format(distribution))
    #
    #     plt.show()

    # build mean over classes
    plt.figure(figsize=(4,4))

    ground_scores = []

    for xai_method in configs["xai_methods"]:

        method_scores = []

        for classidx in classindices:
            try:
                csv = pd.read_csv(resultdir + "_" + xai_method + "_" + distribution + str(classidx) + ".csv")
            except FileNotFoundError:
                csv = pd.read_csv(resultdir + "_" + xai_method + "_" + distribution + "_" + str(classidx) + ".csv")
                if configuration:
                    ref_csv = pd.read_csv("{}_{}_{}_{}.csv".format(ref_resultdir, xai_method, distribution, str(classidx)))
            # print(float(csv["separability_score"]))
            # print(csv)
            percentages = csv["flip_percentage"]
            scores = csv["flipped_score"]

            if configuration:
                scores = scores - ref_csv["flipped_score"]

            method_scores.append(scores)

        method_scores = np.array(method_scores)
        method_scores = np.mean(method_scores, axis=0)

        # if len(ground_scores) == 0:
        #     ground_scores = method_scores
        #
        # method_scores = method_scores - ground_scores

        plt.plot(percentages, method_scores)

    plt.axhline(y=0.0, color='black', linestyle='-', linewidth=1., zorder=0.01)
    plt.xlabel("Ratio of perturbed pixels")
    # plt.ylabel("Score in relation to Saliency")

    if configuration:
        plt.ylabel("Diff. in the Input Perturbation Score")
    else:
        plt.ylabel("Pixelflipping Score")
    # plt.ylim(0.0, 0.7)
    # plt.ylim(0.55, 1.0)
    plt.ylim(-0.135, 0.05)
    plt.grid()

    if log_scale:
        plt.xscale("log")

    # plt.title("Pixelflipping (class-wise mean) with {} replacement".format(distribution))
    plt.legend([LABEL_DICT[x] for x in configs["xai_methods"]])

    # plt.show()
    plt.tight_layout()

    if configuration:
        plt.savefig("../results/canonization/{}_ref_pixelflipping_{}.svg".format(configuration, distribution), format="svg")
    else:
        plt.savefig("../results/figures/canonization/pixelflipping/pixelflipping_{}_start2.svg".format(distribution), format="svg")


# with open(filepath) as file:
#     configs = yaml.load(file, Loader=yaml.FullLoader)
#
#     xai_method = "epsilon_plus"
#     distributions = ["uniform", "gaussian", "inpaint_telea", "inpaint_ns"]
#     resultdir = "../results/pixelflipping"
#
#     resultdir = resultdir + "/" + configs["data"]["dataname"] + "_" + configs["model"]["modelname"]
#
#     log_scale = False
#
#     layers = [configs["layers"][0]]
#     classindices = range(20)
#     # classindices = [2, 4, 6]
#
#     # build mean over classes
#     plt.figure()
#
#     for distribution in distributions:
#
#         method_scores = []
#
#         for classidx in classindices:
#             try:
#                 csv = pd.read_csv(resultdir + "_" + xai_method + "_" + distribution + str(classidx) + ".csv")
#             except FileNotFoundError:
#                 csv = pd.read_csv(resultdir + "_" + xai_method + "_" + distribution + "_" + str(classidx) + ".csv")
#             # print(float(csv["separability_score"]))
#             # print(csv)
#             percentages = csv["flip_percentage"]
#             scores = csv["flipped_score"]
#
#             method_scores.append(scores)
#
#         method_scores = np.array(method_scores)
#         method_scores = np.mean(method_scores, axis=0)
#
#         plt.plot(percentages, method_scores)
#
#     plt.xlabel("percentage of flipped pixels")
#     plt.ylabel("score")
#
#     if log_scale:
#         plt.xscale("log")
#
#     plt.title("Pixelflipping (class-wise mean) with method {}".format(xai_method))
#     plt.legend(distributions)
#
#     plt.tight_layout()
#
#     plt.show()
    # plt.savefig("../results/figures/pixelflipping/auc.svg")
    # plt.savefig("../results/figures/pixelflipping_{}".format(distribution), format="svg")
