import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zennit.image import imgify
from zennit.image import CMAPS

# from ..xaitestframework.helpers.universal_helper import join_path


filepath = "configs/config_experiments.yaml"

options = ["cascading_top_down", "cascading_bottom_up", "independent"]
# option = "independent"

model_result = None
# model_result = "resnet18_voc"

for option in options:

    with open(filepath) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)

        if model_result:
            exampledir = "../results/{}/model_parameter_randomization/{}/explanations".format(model_result, option)
        else:
            exampledir = "../results/model_parameter_randomization/{}/explanations".format(option)

        # layers = [configs["layers"][0]]
        classindices = range(20)

        print(configs["quantifications"])

        for xai_method in configs["xai_methods"]:

            print("visualizations for {}".format(xai_method))
            # for classidx in classindices[1:5]:
            #
            #     path = os.path.join(exampledir, xai_method, str(classidx))
            #
            #     images = [f for f in os.listdir(path) if "{}.npy".format("original") in f]
            #
            #     for image in images:
            #
            #         fname = "_".join(image.split("_")[:2])
            #
            #         fig, ax = plt.subplots(1, len(configs["layers"]) + 1, figsize=(10, 2))
            #
            #         attr = np.load(os.path.join(path, "{}_{}.npy".format(fname, "original")))
            #
            #         attr = np.sum(attr, axis=2)
            #         amax = attr.max((0, 1), keepdims=True)
            #         attr = (attr + amax) / 2 / amax
            #
            #         attr = imgify(attr, vmin=0., vmax=1., level=2., cmap="bwr")
            #
            #         ax[0].imshow(attr)
            #
            #         for l, layer in enumerate(configs["layers"][::-1]):
            #
            #             attr = np.load(os.path.join(path, "{}_{}.npy".format(fname, layer)))
            #
            #             attr = np.sum(attr, axis=2)
            #             amax = attr.max((0, 1), keepdims=True)
            #             attr = (attr + amax) / 2 / amax
            #
            #             attr = imgify(attr, vmin=0., vmax=1., level=2., cmap="bwr")
            #
            #             ax[l+1].imshow(attr)
            #
            #             ax[l+1].set_title(layer)
            #
            #             ax[l+1].set_xticks([])
            #             ax[l+1].set_yticks([])
            #
            #         plt.suptitle(fname)
            #         plt.tight_layout()
            #         plt.show()


            # idx 0
            # 2008_000804   2008_000805

            # idx 1
            # 2008_000194 2008_000803   2008_001225

            # idx 2
            # 2008_000960   2008_000339
            classidx = 0

            fnames = ["2008_000804"]

            if model_result:
                save_path = "../results/{}/model_parameter_randomization/figure_examples/".format(model_result)
            else:
                save_path = "../results/model_parameter_randomization/figure_examples/"

            method_save_path = os.path.join(save_path, xai_method)

            os.makedirs(method_save_path, exist_ok=True)

            path = os.path.join(exampledir, xai_method, str(classidx))

            for fname in fnames:

                attr = np.load(os.path.join(path, "{}_{}.npy".format(fname, "original")))

                if xai_method in ["DeepLift", "GradCam", "GradientXActivation", "IntegratedGradients"]:
                    attr = np.max(attr, axis=2)
                else:
                    attr = np.sum(attr, axis=2)

                amax = attr.max((0, 1), keepdims=True)
                attr = (attr + amax) / 2. / amax

                attr = imgify(attr, vmin=0., vmax=1., level=2., cmap="bwr")

                # save original attribution
                attr.save(os.path.join(method_save_path, "{}_{}.png".format(fname, "original")))

                for l, layer in enumerate(configs["layers"][::-1]):

                    attr = np.load(os.path.join(path, "{}_{}.npy".format(fname, layer)))

                    if xai_method in ["DeepLift", "GradCam", "GradientXActivation", "IntegratedGradients"]:
                        attr = np.max(attr, axis=2)
                    else:
                        attr = np.sum(attr, axis=2)

                    amax = attr.max((0, 1), keepdims=True)
                    attr = (attr + amax) / 2 / amax

                    attr = imgify(attr, vmin=0., vmax=1., level=2., cmap="bwr")

                    # save modified attribution
                    attr.save(os.path.join(method_save_path, "{}_{}_{}.png".format(fname, option, layer)))


            #     layer_scores.append(np.abs(np.array(csv[measure], dtype=float)))   # .as_type(np.float))
            #
            # mean_scores = np.mean(layer_scores, axis=0)
            #
            # plt.plot(csv["layer"], mean_scores)
            #
            # plt.xticks(rotation="45", ha="right")
            # plt.legend(configs["xai_methods"])
            # plt.title("MPR with distance measure {} for option {}".format(measure, option))
            # # plt.show()
            # plt.savefig("../results/figures/mpr/{}_{}".format(measure, option), format="svg")
            # plt.close()
