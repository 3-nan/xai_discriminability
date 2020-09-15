import numpy as np
from sklearn.svm import LinearSVC

from . import innvestigate


def parse_xai_method(xai_method):
    if xai_method == "LRPZ":
        analyzer = innvestigate.analyzer.LRPZ
    elif xai_method == 'LRPEpsilon':
        analyzer = innvestigate.analyzer.LRPEpsilon
    elif xai_method == 'LRPWSquare':
        analyzer = innvestigate.analyzer.LRPWSquare
    elif xai_method == 'LRPAlpha2Beta1':
        analyzer = innvestigate.analyzer.LRPAlpha2Beta1
    elif xai_method == 'LRPSequentialPresetA':
        analyzer = innvestigate.analyzer.LRPSequentialPresetA
    elif xai_method == 'LRPSequentialPresetB':
        analyzer = innvestigate.analyzer.LRPSequentialPresetB
    elif xai_method == 'LRPSequentialCompositeA':
        analyzer = innvestigate.analyzer.LRPSequentialCompositeA
    elif xai_method == 'LRPSequentialCompositeB':
        analyzer = innvestigate.analyzer.LRPSequentialCompositeB
    # innvestigate.analyzer.LRPGamma
    else:
        print("analyzer name " + xai_method + " not correct")
        analyzer = None
    return analyzer


def get_predicted_labels(model, data):
    """ Helper function to compute the predicted labels of the model. """
    y_pred = []

    for batch_data in data.as_numpy_iterator():
        pred = model.predict(batch_data[0])
        y_pred.append(pred)

    y_pred = np.concatenate(y_pred)
    y_pred = np.argmax(y_pred, axis=1)

    return y_pred

def get_relevances_for_layer(input_data, model, layer, xai_method, neuron_selection):
    """ Compute relevance maps for a given model and a specific layer."""

    # remove softmax from model
    analyzer = parse_xai_method(xai_method)

    ana = analyzer(model)

    R = []

    for batch_data in input_data.as_numpy_iterator():
        R_batch = ana.analyze(batch_data[0], neuron_selection=neuron_selection, layer_names=[layer])
        R.append(np.array(R_batch)[0])

    R = np.concatenate(R)

    return R


def compute_separability(R_train, y_train, R_test, y_test):
    """ Computes the separability score for given relevance maps. """
    # reshape / flatten relevance maps

    if len(y_train.shape) == 2:
        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)

    if len(R_train.shape) == 4:
        R_train = R_train.reshape(R_train.shape[0], R_train.shape[1]*R_train.shape[2]*R_train.shape[3])
        R_test = R_test.reshape(R_test.shape[0], R_test.shape[1] * R_test.shape[2] * R_test.shape[3])

    clf = LinearSVC(max_iter=500)
    clf.fit(R_train, y_train)
    result = clf.score(R_test, y_test)

    return result


def evaluate_model_separability(model, train_data, test_data, xai_method, layer):
    """ Compute relevance map separability for each layer of the given model and dataset. """

    # load dataset
    y_train = np.concatenate([ys for imgs, ys in train_data], axis=0)
    y_test = np.concatenate([ys for imgs, ys in test_data], axis=0)

    neuron_selection = "max_activation"

    # get predicted labels
    y_train_pred = get_predicted_labels(model, train_data)
    y_test_pred = get_predicted_labels(model, test_data)

    # prepare model
    model = innvestigate.utils.keras.graph.model_wo_softmax(model)

    R_train = get_relevances_for_layer(train_data, model, layer, xai_method, neuron_selection)
    R_test = get_relevances_for_layer(test_data, model, layer, xai_method, neuron_selection)

    result = compute_separability(R_train, y_train_pred, R_test, y_test_pred)

    return result

'''
def compute_relevances(setup, predicted=True):
    # data
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

    model = keras.models.load_model('../../models/lenet_cifar10/model.h5')

    print(model.metrics_names)
    train_score = model.evaluate(x_train, y_train_categorical, batch_size=64)[1]
    test_score = model.evaluate(x_test, y_test_categorical, batch_size=64)[1]

    predicted_y_train = model.predict(x_train).argmax(axis=-1)
    predicted_y_test = model.predict(x_test).argmax(axis=-1)

    print(model.summary())
    model = innvestigate.utils.model_wo_softmax(model)

    print(setup)
    if predicted:
        #analyzer = innvestigate.create_analyzer(setup, model)
        analyzer = innvestigate.analyzer.LRPAlpha2Beta1(model)
    else:
        #analyzer = innvestigate.create_analyzer(setup, model, neuron_selection_mode="index")
        analyzer = innvestigate.analyzer.LRPAlpha2Beta1(model)

    batchsize = 1000 #125  # 1000
    r_train = []
    masked_train = []

    for i in range(int(x_train.shape[0]/batchsize)):
        print('batch from ' + str(i*batchsize) + 'to' + str((i+1)*batchsize))
        if predicted:
            r_batch = analyzer.analyze(x_train[i*batchsize:(i+1)*batchsize, :, :, :])
        else:
            r_batch = analyzer.analyze(x_train[i * batchsize:(i + 1) * batchsize, :, :, :],
                                       neuron_selection=list(y_train[i * batchsize:(i + 1) * batchsize]))

        r_train.append(r_batch)
        masked_train.append(r_batch*x_train[i*batchsize:(i+1)*batchsize, :, :, :])

    r_train = np.concatenate(r_train, axis=1)
    masked_train = np.concatenate(masked_train, axis=1)
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

    r_test = np.concatenate(r_test, axis=1)

    r_train = r_train.reshape((r_train.shape[1], 1024*3))
    masked_train = masked_train.reshape((masked_train.shape[1], 1024*3))
    masked_test = (r_test * x_test).reshape((r_test.shape[1], 1024*3))

    # save data
    if predicted:
        np.save('../../tmp/r_train_predicted.npy', r_train)
        np.save('../../tmp/r_test_predicted.npy', r_test)
        np.save('../../tmp/masked_train_predicted.npy', masked_train)
        np.save('../../tmp/masked_test_predicted.npy', masked_test)
    else:
        np.save('../../tmp/r_train_true.npy', r_train)
        np.save('../../tmp/r_test_true.npy', r_test)
        np.save('../../tmp/masked_train_true.npy', masked_train)
        np.save('../../tmp/masked_test_true.npy', masked_test)

    np.save('../../tmp/y_train.npy', y_train)
    np.save('../../tmp/y_test.npy', y_test)
    np.save('../../tmp/predicted_y_train.npy', predicted_y_train)
    np.save('../../tmp/predicted_y_test.npy', predicted_y_test)

    return train_score, test_score
    

def svms_on_relevances(predicted=True):

    if predicted:
        r_train = np.load('../../tmp/r_train_predicted.npy')
        r_test = np.load('../../tmp/r_test_predicted.npy')
        masked_train = np.load('../../tmp/masked_train_predicted.npy')
        masked_test = np.load('../../tmp/masked_test_predicted.npy')

    else:
        r_train = np.load('../../tmp/r_train_true.npy')
        r_test = np.load('../../tmp/r_test_true.npy')
        masked_train = np.load('../../tmp/masked_train_true.npy')
        masked_test = np.load('../../tmp/masked_test_true.npy')

    y_train = np.load('../../tmp/y_train.npy')
    y_test = np.load('../../tmp/y_test.npy')
    predicted_y_train = np.load('../../tmp/predicted_y_train.npy')
    predicted_y_test = np.load('../../tmp/predicted_y_test.npy')

    # train svm with rtrain
    print('train relevance svm')
    print(y_train)
    print(y_train.shape)

    clf = LinearSVC(max_iter=200)
    clf.fit(r_train, y_train)

    svm_train_score = clf.score(r_train, y_train)
    print(svm_train_score)

    r_test = r_test.reshape((r_test.shape[1], 1024*3))

    svm_test_score = clf.score(r_test, y_test)
    svm_on_actual_test_score_on_predicted = clf.score(r_test, predicted_y_test)


    # train svm with rtrain on predicted class labels
    print('train relevance svm on predicted class labels')
    clf = LinearSVC(max_iter=200)
    clf.fit(r_train, predicted_y_train)

    svm_on_predicted_train_score = clf.score(r_train, predicted_y_train)
    print(svm_on_predicted_train_score)

    svm_on_predicted_test_score = clf.score(r_test, predicted_y_test)
    svm_on_predicted_test_score_on_actual = clf.score(r_test, y_test)

    # train svm on masked cifar10
    print('train relevance-masked data svm')

    clf = LinearSVC(max_iter=200)
    clf.fit(masked_train, y_train)

    svm_masked_train_score = clf.score(masked_train, y_train)

    # test model on masked r_test
    svm_masked_test_score = clf.score(masked_test, y_test)

    return svm_train_score, svm_test_score, svm_on_actual_test_score_on_predicted, \
           svm_on_predicted_train_score, svm_on_predicted_test_score, svm_on_predicted_test_score_on_actual, \
           svm_masked_train_score, svm_masked_test_score


results = []

setups = ['test']
        #'gradient', 'smoothgrad', 'lrp.z',
         # 'lrp.epsilon',
          #'lrp.w_square',
          #'lrp.alpha_2_beta_1',
          #'lrp.alpha_1_beta_0',
          #'lrp.z_plus',
          #'lrp.z_plus_fast',
          #'lrp.sequential_preset_a',
          #'lrp.sequential_preset_b']

for setup in setups:
    train_score, test_score = compute_relevances(setup, predicted=False)
    _, _ = compute_relevances(setup, predicted=True)

    svm_train_score, svm_test_score, svm_on_actual_test_score_on_predicted, svm_on_predicted_train_score, \
        svm_on_predicted_test_score, svm_on_predicted_test_score_on_actual, svm_masked_train_score, \
        svm_masked_test_score = svms_on_relevances(predicted=False)

    psvm_train_score, psvm_test_score, psvm_on_actual_test_score_on_predicted, psvm_on_predicted_train_score, \
    psvm_on_predicted_test_score, psvm_on_predicted_test_score_on_actual, psvm_masked_train_score, \
    psvm_masked_test_score = svms_on_relevances(predicted=True)

    # append results
    results.append(['LeNet5', setup, train_score, test_score, svm_train_score, svm_test_score,
                    svm_on_actual_test_score_on_predicted,
                    svm_on_predicted_train_score, svm_on_predicted_test_score, svm_on_predicted_test_score_on_actual,
                    svm_masked_train_score, svm_masked_test_score,
                    psvm_train_score, psvm_test_score, psvm_on_actual_test_score_on_predicted,
                    psvm_on_predicted_train_score, psvm_on_predicted_test_score, psvm_on_predicted_test_score_on_actual,
                    psvm_masked_train_score, psvm_masked_test_score])

    df = pd.DataFrame(results, columns=['model', 'lrp_setup', 'nn_train_acc', 'nn_test_acc',
                                        'svm_train_score', 'svm_test_score', 'svm_on_actual_test_score_on_predicted',
                                        'svm_on_predicted_labels_train_score', 'svm_on_predicted_labels_test_score',
                                        'svm_on_predicted_test_scor e_on_actual',
                                        'svm_masked_train_score', 'svm_masked_test_score',
                                        'psvm_train_score', 'psvm_test_score', 'psvm_on_actual_test_score_on_predicted',
                                        'psvm_on_predicted_train_score', 'psvm_on_predicted_test_score',
                                        'psvm_on_predicted_test_score_on_actual',
                                        'psvm_masked_train_score', 'psvm_masked_test_score'])
    df.to_csv('results_cifar10_' + 'LeNet5' + '_linearsvm_smoothgrad_v3' + '.csv', index=False)
'''
