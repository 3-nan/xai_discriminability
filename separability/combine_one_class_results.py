import glob
import pandas as pd

setup = "imagenet_vgg16"

data_path = 'results/one_class/'

methods = ['Gradient', 'SmoothGrad', 'LRPZ', 'LRPAlpha1Beta0', 'LRPGamma', 'LRPSequentialCompositeA', 'LRPSequentialCompositeBFlat']
# layers = ['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_4', 'conv2d_7', 'conv2d_10', 'dense', 'dense_1', 'dense_2']
layers = ['input_1', 'block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1', 'fc1', 'fc2']

files = []

for method in methods:
    for layer in layers:
        csv_files = glob.glob(data_path + setup + '/' + layer + '_' + method + '*.csv')
        if csv_files:
            sum = 0
            for f in csv_files:
                csv = pd.read_csv(f)
                sum += float(csv.values[0][-1])
            sum = sum / len(csv_files)
            print(sum)
            comb = pd.read_csv(csv_files[0])
            comb.values[0][-1] = str(sum)
            # comb = pd.concat([pd.read_csv(f) for f in csv_files], axis=1)
            files.append(comb)
        else:
            print('No files found for layer ' + layer + ' with rule ' + method)

# files = [file for file in glob.glob('results/vgg16/cifar10/' + method + '/*/job_results/*.csv')]

combined_csv = pd.concat(files, axis=0)
#combined_csv = pd.concat([pd.read_csv(f) for f in files], axis=0)
combined_csv.to_csv("results/one_class_" + setup + "_combined.csv", index=False)
