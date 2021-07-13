import os
import yaml
import numpy as np
import matplotlib.pyplot as plt


filepath = "configs/config_experiments.yaml"

with open(filepath) as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)

    distributions = ["uniform", "gaussian", "inpaint_telea", "inpaint_ns"]
    # cmap = plt.cm.get_cmap("Set1", len(distributions))
    layer = "linear2"
    # distribution = "uniform"

    for distribution in distributions:

        resultdir = "../results/tsne_manifold_one_class/Saliency/{}/{}".format(layer, distribution)

        args = configs["quantifications"][0]["tsne_manifold_one_class"]["args"]
        print(args["percentages"])

        train_embedding = np.load(os.path.join(resultdir, "tsne_embedding.npy"))
        targets = np.load(os.path.join(resultdir, "targets.npy"))

        print(train_embedding.shape)
        print(targets.shape)
        print(np.unique(targets))

        unique, counts = np.unique(targets, return_counts=True)
        print(counts)

        sample_points = []

        percentages = [0.0, 0.002, 0.02, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

        # new code
        plt.figure()
        scat = plt.scatter(train_embedding[268:, 0], train_embedding[268:, 1], s=5, c=targets[268:])
        # for l in range(0, counts[-1], 10):
        #     plt.plot(train_embedding[(269+l)::258, 0], train_embedding[(269+l)::258, 1], c="gray", alpha=0.4)
        plt.legend(handles=scat.legend_elements()[0], labels=percentages)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()
        # plt.savefig("../results/figures/tsne_manifold_one_class/{}_{}.svg".format(layer, distribution))
        plt.close()
    # plt.savefig(os.path.join(resultdir, "tsne_manifold_one_class_{}.svg".format(distribution)))
    # end new code

    # for percentage in percentages:
    #     new_points = np.load(os.path.join(resultdir, "15", "percentage_{}.npy".format(percentage)), allow_pickle=True)
    #     new_points = np.vstack(new_points)      # TODO ist falsch!!!
    # #
    # #     sample_points.append(new_points[10, :])
    #
    #     new_points = np.array(new_points)
    #     print(np.shape(new_points))
    #
    #     plt.figure()
    #     # plt.scatter(sample_points[:, 0], sample_points[:, 1], s=5.)
    #     plt.scatter(train_embedding[:, 0], train_embedding[:, 1], s=3., c=targets, cmap="Set1")
    #     plt.scatter(new_points[:, 0], new_points[:, 1], s=3.)
    #     # plt.scatter(new_points, s=3)
    #     plt.show()
