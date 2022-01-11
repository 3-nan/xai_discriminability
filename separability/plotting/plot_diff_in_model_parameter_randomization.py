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

    print(configs["quantifications"])

    for measure in configs["quantifications"][0]["model_parameter_randomization"]["args"]["distance_measures"]:

        cmap = plt.cm.get_cmap("Paired", 8)

        plt.figure()

        for xai_method in configs["xai_methods"]:

            layer_scores = []
            ref_layer_scores = []

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

                layer_scores.append(score)
                ref_layer_scores.append(ref_score)
                # layer_scores.append(score - ref_score)   # .as_type(np.float))

            mean_scores = np.mean(layer_scores, axis=0)
            ref_mean_scores = np.mean(ref_layer_scores, axis=0)

            ind = np.arange(len(csv["layer"]))

            plt.bar(ind - 0.2, mean_scores, 0.4, label="canonized", color=cmap(0))
            plt.bar(ind + 0.2, ref_mean_scores, 0.4, label="uncanonized", color=cmap(1))

        plt.xticks(ind, csv["layer"])
        plt.ylabel("Score")
        plt.xticks(rotation="45", ha="right")
        # plt.ylim(-0.1, 1.1)
        plt.legend(["canonized", "uncanonized"])

        if option == "cascading_top_down":
            plt.title("Top-down cascading MPR with distance measure {}".format(measure))
        elif option == "cascading_bottom_up":
            plt.title("Bottom-up cascading MPR with distance measure {}".format(measure))
        elif option == "independent":
            plt.title("Independent MPR with distance measure {}".format(measure))
        # plt.show()
        plt.savefig("../results/figures/{}/mpr/diff_{}_{}_ref.svg".format(configuration, measure, option), format="svg")
        plt.close()
