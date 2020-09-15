import tensorflow as tf


def get_model(modelname):

    if modelname == "vgg16":

        model = tf.keras.applications.VGG16(
            include_top=True,
            weights="imagenet",
            classifier_activation="softmax"
        )

    return model
