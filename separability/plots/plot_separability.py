from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

#file = glob('results/one_class_results')
#print(file)
csv = pd.read_csv("results/one_class_combined.csv", index_col=[0, 1, 2, 3])
print(csv)
print(csv.values)
print(csv[np.in1d(csv.index.get_level_values(3), ['Gradient'])])

print(csv.index.get_level_values(3))
print(csv.index.unique(level=3))

layers = ['conv2d', 'conv2d_2', 'conv2d_4', 'conv2d_7', 'conv2d_10', 'dense', 'dense_1', 'dense_2']

plt.figure()

for method in csv.index.unique(level=3):
    results = csv[np.in1d(csv.index.get_level_values(3), [method])]
    # add missing data to dataframe
    values = []
    for layer in layers:
        if layer in results.index.get_level_values(2):
            values.append(results.loc['cifar10', 'vgg16', layer, method][-1])
        else:
            values.append(np.nan)

    print(type(values))
    print(values)
    values = np.array(values)
    nans, x = nan_helper(values)
    values[nans] = np.interp(x(nans), x(~nans), values[~nans])
    print(results)
    plt.plot(layers, values, linestyle="--", marker="o")

plt.ylim([0.5, 1.02])
plt.ylabel("Separability score")
plt.xlabel("Layer")
plt.title("Accumulated one-class separability")
plt.legend(csv.index.unique(level=3))

plt.show()
