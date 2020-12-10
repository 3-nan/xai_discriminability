import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]


def reformat_string(x):
    for symbol in ["\n", "[", "]", ","]:
        x = x.replace(symbol, "")

    print(x)
    x = x.split(" ")
    print(x)

    x = [np.float(number) for number in x]
    # x = np.fromstring(x, dtype=np.float)
    return x

setup = "imagenet_vgg16"
filepath = "results/flipping/descending/" # + setup + "_combined.csv"

plt.figure()

methods = []

for root, _, files in os.walk(filepath):

    for file in files:
        print(file)

        csv = pd.read_csv(root+file, header=0, index_col=False) # , index_col=[0, 1, 2])
        scores = []
        for x in csv["flipped_score"]:
            scores.append(reformat_string(x))

        scores = np.array(scores)

        plt.plot(csv["flip_percentage"], scores[:, 1]) #, linestyle="--", marker="o")

        methods.append(csv["method"][0])

plt.xlabel("Flip Percentage")
plt.ylabel("Classification Accuracy")
plt.legend(methods)

plt.show()
'''

plt.figure()

for method in csv.index.unique(level=3):
    results = csv[np.in1d(csv.index.get_level_values(3), [method])]
    # add missing data to dataframe
    values = []
    for layer in layers:
        if layer in results.index.get_level_values(2):
            values.append(results.loc['imagenet', 'vgg16', layer, method][-1])
        else:
            values.append(np.nan)

    print(type(values))
    print(values)
    values = np.array(values)
    nans, x = nan_helper(values)
    values[nans] = np.interp(x(nans), x(~nans), values[~nans])
    print(results)
    plt.plot(layers, values, linestyle="--", marker="o")

plt.ylim([0.7, 1.02])
plt.ylabel("Separability score")
plt.xlabel("Layer")
plt.title("Accumulated one-class separability")
plt.legend(csv.index.unique(level=3))

plt.show()
'''