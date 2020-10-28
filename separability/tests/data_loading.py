import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

for c in range(10):
    indices = (y_train == c).flatten()
    print(len(indices))
    print(indices[:3])
    print(indices.shape)
    print((x_train[indices]).shape)