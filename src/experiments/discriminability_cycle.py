import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10

import innvestigate


def compute_relevance_maps(model_path, setup):
    """ Load model and compute relevance maps:
    INPUT:
        model_path - path to model to use for relevance computation

    """
    # load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Convert class vectors to binary class matrices.
    y_train_categorical = keras.utils.to_categorical(y_train, 10)
    y_test_categorical = keras.utils.to_categorical(y_test, 10)

    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalize inputs
    x_train = x_train / 127.5 - 1
    x_test = x_test / 127.5 - 1

    model = keras.models.load_model(model_path)

    print(model.metrics_names)
    train_score = model.evaluate(x_train, y_train_categorical, batch_size=64)[1]
    test_score = model.evaluate(x_test, y_test_categorical, batch_size=64)[1]

    predicted_y_train = model.predict(x_train).argmax(axis=-1)
    predicted_y_test = model.predict(x_test).argmax(axis=-1)

    print(model.summary())
    model = innvestigate.utils.model_wo_softmax(model)

    print(setup)
    if predicted:
        analyzer = innvestigate.create_analyzer(setup, model)
        # analyzer = innvestigate.analyzer.LRPAlpha2Beta1_new(model)
    else:
        analyzer = innvestigate.create_analyzer(setup, model, neuron_selection_mode="index")
        # analyzer = innvestigate.analyzer.LRPAlpha2Beta1_new(model)

    batchsize = 1000  # 125  # 1000
    r_train = []

    for i in range(int(x_train.shape[0] / batchsize)):
        print('batch from ' + str(i * batchsize) + 'to' + str((i + 1) * batchsize))
        if predicted:
            r_batch = analyzer.analyze(x_train[i * batchsize:(i + 1) * batchsize, :, :, :])
        else:
            r_batch = analyzer.analyze(x_train[i * batchsize:(i + 1) * batchsize, :, :, :],
                                       neuron_selection=list(y_train[i * batchsize:(i + 1) * batchsize]))

        r_train.append(r_batch)

    r_train = np.concatenate(r_train)
    print(r_train.shape)

    # test relevances
    r_test = []
    for i in range(int(x_test.shape[0] / batchsize)):
        if predicted:
            r_batch = analyzer.analyze(x_test[i * batchsize:(i + 1) * batchsize, :, :, :])
        else:
            r_batch = analyzer.analyze(x_test[i * batchsize:(i + 1) * batchsize, :, :, :],
                                       neuron_selection=y_test[i * batchsize:(i + 1) * batchsize].tolist())

        r_test.append(r_batch)

    r_test = np.concatenate(r_test)

    r_train = r_train.reshape((r_train.shape[0], 1024 * 3))

    # save data
    if predicted:
        np.save('tmp/r_train_predicted.npy', r_train)
        np.save('tmp/r_test_predicted.npy', r_test)

    else:
        np.save('tmp/r_train_true.npy', r_train)
        np.save('tmp/r_test_true.npy', r_test)

    np.save('tmp/y_train.npy', y_train)
    np.save('tmp/y_test.npy', y_test)
    np.save('tmp/predicted_y_train.npy', predicted_y_train)
    np.save('tmp/predicted_y_test.npy', predicted_y_test)

    return train_score, test_score


def compute_discriminability():
    """ trains and tests an svm on computed relevance maps. Saves svm performance to csv."""