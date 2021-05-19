import os
import yaml
import numpy as np
import matplotlib.pyplot as plt


filepath = "configs/config_experiments.yaml"

with open(filepath) as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)

    distributions = ["uniform", "gaussian", "inpaint_telea", "inpaint_ns"]
    # cmap = plt.cm.get_cmap("Set1", len(distributions))

    resultdir = "../results/tsne_manifold"

    args = configs["quantifications"][0]["tsne_manifold"]["args"]
    print(args["percentages"])

    train_embedding = np.load(os.path.join(resultdir, "tsne_embedding_layer2_7.npy"))
    targets = np.load(os.path.join(resultdir, "targets_layer2_7.npy"))

    print(train_embedding.shape)
    print(targets.shape)
    print(np.unique(targets))

    # sample_points = []
    #
    # percentages = [0.0, 0.0002, 0.002, 0.02, 0.5]
    #
    # for percentage in percentages:
    #     new_points = np.load(os.path.join(resultdir, "2", "percentage_{}.npy".format(percentage)), allow_pickle=True)
    #     new_points = np.vstack(new_points)      # TODO ist falsch!!!
    #
    #     sample_points.append(new_points[10, :])

    # sample_points = np.array(sample_points)

    plt.figure()
    # plt.scatter(sample_points[:, 0], sample_points[:, 1], s=5.)
    plt.scatter(train_embedding[:, 0], train_embedding[:, 1], s=3., c=targets, cmap="Set1")
    # plt.scatter(new_points[0][:, 0], new_points[0][:, 1], s=3.)
    # plt.scatter(new_points, s=3)
    plt.show()
