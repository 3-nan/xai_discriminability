import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from ..xaitestframework.helpers.universal_helper import join_path


filepath = "configs/config_experiments.yaml"

dataname = "imagenet"
dconfig = "imagenet"
modelname = "resnet18"   # "resnet18"    vgg16bn

configuration = "{}_{}".format(modelname, dconfig)
ref_configuration = "{}_{}_uncanonized".format(modelname, dconfig)

option = "independent"      # "cascading_top_down"  "independent" "cascading_bottom_up"

if dataname == "VOC2012":
    classindices = range(20)
elif dataname == "imagenet":
    classindices = [96, 126, 155, 292, 301, 347, 387, 405, 417, 426,
                    446, 546, 565, 573, 604, 758, 844, 890, 937, 954]
else:
    print("please specify classindices for dataname {}!".format(dataname))


with open(filepath) as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)

    resultdir = "../results/{}/model_parameter_randomization/{}/{}_{}"\
        .format(configuration, option, dataname, modelname)

    ref_resultdir = "../results/{}/model_parameter_randomization/{}/{}_{}"\
        .format(ref_configuration, option, dataname, modelname)

    # plt.figure()
    #
    # for xai_method in configs["xai_methods"]:
    #
    #     method_scores = []
    #
    #     for classidx in classindices:
    #
    #         try:
    #             csv = pd.read_csv(resultdir + "_" + xai_method + "_" + str(classidx) + ".csv")
    #
    #             method_scores.append(csv["score"])
    #
    #         except FileNotFoundError:
    #             print("file for " + xai_method + " with idx " + str(classidx) + " not found")
    #
    #     method_scores = np.array(method_scores)
    #     method_scores = np.mean(method_scores, axis=0)
    #
    #     plt.plot(csv["layer"], method_scores)
    #
    # plt.legend(configs["xai_methods"])
    # plt.show()
    #
    # for classidx in classindices:
    #
    #     plt.figure()
    #
    #     for xai_method in configs["xai_methods"]:
    #
    #         print(xai_method)
    #         # scores = []
    #
    #         csv = pd.read_csv(resultdir + "_" + xai_method + "_" + str(classidx) + ".csv")
    #
    #         # scores.append(float(csv["separability_score"]))
    #
    #         plt.plot(csv["layer"], csv["score"])
    #
    #     plt.xlabel("layer")
    #     plt.ylabel("score")
    #     plt.legend(configs["xai_methods"])
    #     plt.title("MPR mean difference (mse) of explanations for class {}".format(classidx))
    #     plt.show()

    print(configs["quantifications"])

    for measure in configs["quantifications"][0]["model_parameter_randomization"]["args"]["distance_measures"]:

        plt.figure()

        for xai_method in configs["xai_methods"]:

            layer_scores = []

            for classidx in classindices:

                csv = pd.read_csv(resultdir + "_" + xai_method + "_" + str(classidx) + ".csv")
                ref_csv = pd.read_csv("{}_{}_{}.csv".format(ref_resultdir, xai_method, str(classidx)))

                score = np.abs(np.array(csv[measure], dtype=float))
                ref_score = np.abs(np.array(ref_csv[measure], dtype=float))

                if np.sum(score-ref_score) != 0.:
                    print("{} {} {}".format(measure, xai_method, classidx))

                nan_indices = np.isnan(score)
                nan_indices_ref = np.isnan(ref_score)

                if nan_indices.any() or nan_indices_ref.any():

                    nans = np.logical_or(nan_indices, nan_indices_ref)

                    score[nans] = 0.
                    ref_score[nans] = 0.

                # layer_scores.append(score)
                layer_scores.append(score - ref_score)   # .as_type(np.float))

            mean_scores = np.mean(layer_scores, axis=0)

            plt.plot(csv["layer"], mean_scores)

        plt.axhline(y=0.0, color='black', linestyle='-', linewidth=2., zorder=0.01)
        plt.ylabel("Diff. to uncanonized baseline")
        plt.xticks(rotation="45", ha="right")
        # plt.ylim(-0.1, 1.1)
        plt.legend(configs["xai_methods"])

        if option == "cascading_top_down":
            plt.title("Top-down cascading MPR with distance measure {}".format(measure))
        elif option == "cascading_bottom_up":
            plt.title("Bottom-up cascading MPR with distance measure {}".format(measure))
        elif option == "independent":
            plt.title("Independent MPR with distance measure {}".format(measure))
        # plt.show()
        plt.savefig("../results/figures/{}/mpr/{}_{}_ref.svg".format(configuration, measure, option), format="svg")
        plt.close()
