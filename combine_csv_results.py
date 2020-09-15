import glob
import pandas as pd
from tensorflow import keras

method = 'LRPCMPA'
model_path = 'models/vgg16_cifar10/model'

model = keras.models.load_model(model_path)

files = []

for layer in model.layers:
    csv_file = glob.glob('results/vgg16/cifar10/' + method + '/' + layer.name + '/job_results/*.csv')
    if csv_file:
        files.append(csv_file[0])
    else:
        print('No file found for layer ' + layer.name)

# files = [file for file in glob.glob('results/vgg16/cifar10/' + method + '/*/job_results/*.csv')]

combined_csv = pd.concat([pd.read_csv(f) for f in files], axis=1)
combined_csv.to_csv(method + ".csv", index=False)
