import glob
import pandas as pd

data_path = 'results/'

methods = ['Gradient', 'LRPZ', 'LRPAlpha1Beta0', 'LRPGamma', 'LRPSequentialCompositeA', 'LRPSequentialCompositeBFlat']
layers = ['conv2d', 'conv2d_2', 'conv2d_4', 'conv2d_7', 'conv2d_10', 'dense', 'dense_1', 'dense_2']

files = []

for method in methods:
    for layer in layers:
        csv_file = glob.glob(data_path + 'cifar10_vgg16_' + layer + '_' + method + '.csv')
        if csv_file:
            files.append(csv_file[0])
        else:
            print('No file found for layer ' + layer + ' with rule ' + method)

# files = [file for file in glob.glob('results/vgg16/cifar10/' + method + '/*/job_results/*.csv')]

combined_csv = pd.concat([pd.read_csv(f) for f in files], axis=0)
combined_csv.to_csv("results/combined.csv", index=False)
