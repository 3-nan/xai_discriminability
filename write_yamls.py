import yaml
import tensorflow.keras as keras

model = keras.models.load_model('train_models/train_vgg16_cifar10/model')

layers = []

for layer in model.layers:
    layers.append(layer.name)

for layername in layers:
    dict_file = {
        'dataset_path': '/mnt/dataset/',
        'dataset': 'cifar10',
        'model_path': '/mnt/additional_files/model/',
        'model_name': 'cifar10_vgg16',
        'xai_method': 'LRPAlpha1Beta0',
        'layer': layername
    }

    with open('pipeline/additional_files/parameters/' + layername + '.yaml', 'w') as file:
        documents = yaml.dump(dict_file, file)
