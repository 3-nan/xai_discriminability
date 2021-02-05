import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2


def gregoire_black_firered(R, normalize=True):

    if normalize:
        R /= np.max(np.abs(R))

    x = R

    hrp = np.clip(x-0.00,0,0.25)/0.25
    hgp = np.clip(x-0.25,0,0.25)/0.25
    hbp = np.clip(x-0.50,0,0.50)/0.50

    hbn = np.clip(-x - 0.00, 0, 0.25) / 0.25
    hgn = np.clip(-x - 0.25, 0, 0.25) / 0.25
    hrn = np.clip(-x - 0.50, 0, 0.50) / 0.50

    return np.concatenate([(hrp+hrn)[...,None], (hgp+hgn)[...,None],(hbp+hbn)[...,None]], axis=2)


def prep_rmap(R):
    R = R.sum(axis=np.argmax(np.asarray(R.shape) == 3))
    return gregoire_black_firered(R)


def rebuild_original(img):
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68

    img[img > 255] = 255
    img = img.astype(np.uint8)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def plot_model_parameter_randomization_example(modelpath, setup, dir, index, independent=False, bottom_up=False):

    print(dir + setup)

    model = tf.keras.models.load_model(modelpath)
    layers = [layer.name for layer in model.layers if hasattr(layer, 'kernel_initializer')]
    print(layers)

    original = np.load(dir + setup + "/" + "original.npy")[index]
    relevance = np.load(dir + setup + "/not_randomized.npy")[index]

    fig, axs = plt.subplots(4, 5, figsize=(12, 12), sharex=True, sharey=True)
    axs[0, 0].imshow(rebuild_original(original))
    axs[0, 0].set_title("original")
    axs[0, 1].imshow(prep_rmap(relevance))
    axs[0, 1].set_title("none")

    x = 2
    y = 0

    for layer in layers:
        if independent:
            filepath = dir + setup + "/independent/" + layer + ".npy"
        elif bottom_up:
            filepath = dir + setup + "/cascading/bottom_up/" + layer + ".npy"
        else:
            filepath = dir + setup + "/cascading/" + layer + ".npy"

        try:
            rmap = np.load(filepath)[index]
            axs[y, x].imshow(prep_rmap(rmap))
            axs[y, x].set_title(layer)
            if x == 4:
                y += 1
                x = 0
            else:
                x += 1
        except IOError:
            print("No file for layer {} found".format(layer))

    plt.tight_layout()
    plt.show()

    if independent:
        fig.savefig("plots2020/model_parameter_randomization/independent/" + setup + ".png")
    elif bottom_up:
        fig.savefig("plots2020/model_parameter_randomization/cascading_bottomup/" + setup + ".png")
    else:
        fig.savefig("plots2020/model_parameter_randomization/cascading/" + setup + ".png")


modelpath = "../models/vgg16_imagenet/"
setups = ["imagenet_vgg16_LRPGamma", "imagenet_vgg16_LRPAlpha1Beta0", "imagenet_vgg16_LRPZ",
          "imagenet_vgg16_LRPSequentialCompositeA", "imagenet_vgg16_LRPSequentialCompositeBFlat",
          "imagenet_vgg16_Gradient", "imagenet_vgg16_SmoothGrad"]
dir = "results/model_parameter_randomization/"
index = 45
independent = False
bottom_up = False

for setup in setups:
    plot_model_parameter_randomization_example(modelpath, setup, dir, index, independent, bottom_up)