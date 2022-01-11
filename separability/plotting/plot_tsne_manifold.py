import os
import yaml
import numpy as np
import matplotlib.pyplot as plt


filepath = "configs/config_experiments.yaml"

with open(filepath) as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)

    distributions = ["uniform", "gaussian", "inpaint_telea", "inpaint_ns"]
    # cmap = plt.cm.get_cmap("Set1", len(distributions))

    # distribution = "gaussian"
    for distribution in distributions:
        layer = "conv13"

        resultdir = "../results/tsne_manifold/Saliency/{}/{}".format(layer, distribution)

        args = configs["quantifications"][0]["tsne_manifold"]["args"]
        print(args["percentages"])

        train_embedding = np.load(os.path.join(resultdir, "tsne_embedding_30.npy"))   # tsne_embedding_layer2_7
        targets = np.load(os.path.join(resultdir, "targets_30.npy"))  # targets_layer2_7

        print(train_embedding.shape)
        print(targets.shape)
        print(np.unique(targets))

        # sample_points = []

        # classindices = ["2", "4", "6"]
        classindices = ["3", "8", "12"]

        # percentages = [0.0, 0.0002, 0.0004, 0.0006, 0.0008, 0.002, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

        percentages = [0.0, 0.02, 0.2, 0.5, 0.9]

        # plt.figure()
        # plt.scatter(train_embedding[:, 0], train_embedding[:, 1], s=3., c=targets, cmap="Set1")
        # plt.show()

        fig, ax = plt.subplots(len(classindices), len(percentages), figsize=(15, 5))

        for c, classidx in enumerate(classindices):
            for p, percentage in enumerate(percentages):
                new_points = np.load(os.path.join(resultdir, classidx, "percentage_{}.npy".format(percentage)), allow_pickle=True)
                new_points = np.vstack(new_points)      # TODO ist falsch!!!

                # sample_points.append(new_points[10, :])

                sample_points = np.array(new_points)

                # plt.figure()
                ax[c][p].scatter(train_embedding[:, 0], train_embedding[:, 1], s=3., c=targets, cmap="Set1")
                ax[c][p].scatter(sample_points[:, 0], sample_points[:, 1], s=3.)
                # plt.scatter(new_points[0][:, 0], new_points[0][:, 1], s=3.)
                # plt.scatter(new_points, s=3)

            for axes in ax[c]:
                axes.set_xticks([])
                axes.set_yticks([])
            ax[c][0].set_ylabel("class {}".format(classidx))
        for p, percentage in enumerate(percentages):
            ax[0][p].set_xlabel(percentage)
            ax[0][p].xaxis.set_label_position("top")

        fig.suptitle("Distribution of flipped images within original tSNE Embedding: {}".format(distribution))
        plt.tight_layout()
        plt.show()
        # plt.savefig(os.path.join(resultdir, "tsne_flipping_{}.png".format(distribution)))
        # plt.savefig("../results/figures/tsne_manifold/{}_{}.svg".format(layer, distribution))

        for c, classidx in enumerate(classindices):
            for p, percentage in enumerate(percentages):
                new_points = np.load(os.path.join(resultdir, classidx, "percentage_{}.npy".format(percentage)), allow_pickle=True)
                new_points = np.vstack(new_points)      # TODO ist falsch!!!

                # sample_points.append(new_points[10, :])

                sample_points = np.array(new_points)

                plt.figure()
                plt.scatter(train_embedding[:, 0], train_embedding[:, 1], s=70., c=targets, cmap="Set1")
                plt.scatter(sample_points[:, 0], sample_points[:, 1], s=70.)
                # plt.scatter(new_points[0][:, 0], new_points[0][:, 1], s=3.)
                # plt.scatter(new_points, s=3)

                plt.xticks([])
                plt.yticks([])

                plt.tight_layout()
                # plt.savefig("../results/figures/tsne_manifold/{}/{}_{}_{}.svg".format(layer, distribution, classidx, percentage), bbox_inches='tight')

                # plt.show()
                # plt.waitforbuttonpress()
                plt.close()
