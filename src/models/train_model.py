import numpy as np
#import importlib.util as imp
#if imp.find_spec("cupy"): #use cupy for GPU support if available
#    import cupy as np
import cupy as cp
from glob import glob
from sklearn.svm import SVC

from lrp_toolbox.python import data_io


def test_svm(nn, lrp_version):
    """
    params: nn - name of the neural network used
            lrp_version - version of the lrp algorithm
    """
    X = []
    count = len(glob('../../data/' + nn + '/Rtrain_' + lrp_version + '_*'))
    print(count)

    for i in range(count):
        x = data_io.read('../../data/' + nn + '/Rtrain_' + lrp_version + '_' + str(i) + '.npy')
        X.append(x)

    X = np.concatenate(X)
    Y = data_io.read('../../lrp_toolbox/data/MNIST/train_labels.npy')

    X = cp.asnumpy(X)
    Y = cp.asnumpy(Y)

    print(X.shape)

    svc = SVC()
    batch_size = 10000

    for i in np.arange(1, int(X.shape[0] / batch_size) + 1):
        x_batch = X[((i - 1) * batch_size):(i * batch_size), :]
        y_batch = Y[((i - 1) * batch_size):(i * batch_size), :]
        svc = svc.fit(x_batch, np.ravel(y_batch))
    print('training completed')

    train_score = svc.score(X, np.ravel(Y))
    print(train_score)

    X = []
    count = len(glob('../../data/' + nn + '/Rtest_' + lrp_version + '_*'))
    print(count)

    for i in range(count):
        x = data_io.read('../../data/' + nn + '/Rtest_' + lrp_version + '_' + str(i) + '.npy')
        X.append(x)

    X = np.concatenate(X)
    Y = data_io.read('../../lrp_toolbox/data/MNIST/test_labels.npy')

    X = cp.asnumpy(X)
    Y = cp.asnumpy(Y)

    test_score = svc.score(X, np.ravel(Y))
    print(test_score)

    return train_score, test_score


# test_svm('long_tanh', 'simple')

