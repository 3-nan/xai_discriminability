import os
import yaml
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set_theme(style="darkgrid")
# sns.set_palette(sns.color_palette("hls", 20))

filepath = "configs/config_experiments.yaml"

with open(filepath) as file:
    configs = yaml.load(file, Loader=yaml.FullLoader)

    distributions = ["uniform", "gaussian", "inpaint_telea", "inpaint_ns"]

    example_dir = "../results/pixelflipping/examples"

    percentages = configs["quantifications"][0]["pixelflipping"]["args"]["percentages"]
    print(percentages)

    layers = [configs["layers"][0]]
    classidx = 0
    sample_name = "2008_000021"

    for xai_method in configs["xai_methods"]:

        for distribution in distributions:
            sample_dir = os.path.join(example_dir, xai_method, str(classidx), distribution)

            for percentage in percentages:

                sample = os.path.join(sample_dir, "{}_{}.npy".format(sample_name, percentage))
            # read files
            # samples = glob.glob(os.path.join(sample_dir, "*0.0.npy"))

            # for sample in samples:

                # sample_path = "_".join(sample.split("_")[:-1])
                #
                # fig = plt.figure(figsize=(18, 7))
                # rows = 3
                # columns = 8
                #
                # for p, percentage in enumerate(percentages):
                img = np.load(sample)

                # scale to 1
                img += 1.
                img /= 2.

                if np.min(img) < 0.:
                    print(np.min(img))

                if np.max(img) > 1.:
                    print(np.max(img))

                img = img[:, :, ::-1]

                    # fig.add_subplot(rows, columns, p + 1)
                plt.figure()
                plt.axis("off")
                plt.imshow(img)
                plt.savefig("../results/pixelflipping_example/{}_{}_{}.svg".format(sample_name, distribution, percentage), bbox_inches='tight')
