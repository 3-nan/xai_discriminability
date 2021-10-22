import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("/home/motzkus/work/xai_discriminability/separability")
from xaitestframework.dataloading.custom import VOC2012Dataset, MyImagenetDataset
from xaitestframework.dataloading.dataloader import DataLoader
from xaitestframework.models.pytorchmodel import PytorchModel


def simple_sampling(batch, indicesfraction, flipping_method):
    flipped_data = []

    # flip images
    for s, sample in enumerate(batch):
        # flip pixels

        datum = sample.datum
        for axis in range(datum.shape[2]):
            if flipping_method == "uniform":
                random_values = np.random.uniform(-1.0, 1.0, len(indicesfraction[s]))
            elif flipping_method == "gaussian":
                random_values = np.random.normal(loc=0.0, scale=1.0, size=len(indicesfraction[s]))
            else:
                raise ValueError("No distribution for flipping pixels specified.")
            np.put_along_axis(datum[:, :, axis], indicesfraction[s], random_values, axis=None)

        flipped_data.append(datum)

    flipped_data = np.array(flipped_data)
    return flipped_data


def inpainting(batch, indicesfraction, flipping_method):
    flipped_data = []

    for s, sample in enumerate(batch):
        # build mask
        mask = np.zeros(sample.datum.shape[:2], dtype=np.uint8)
        np.put_along_axis(mask, indicesfraction[s], 1.0, axis=None)

        # get filepath
        # filepath = "not implemented"  # ToDo: how to get filepath/image file as expected?
        # img = cv2.imread(filepath, cv2.IMREAD_COLOR)

        datum = sample.datum

        datum = datum.astype(np.float32)

        if flipping_method == "inpaint_telea":
            # sample.filename
            for channel in range(datum.shape[2]):
                datum[:, :, channel] = cv2.inpaint(datum[:, :, channel], mask, 3, cv2.INPAINT_TELEA)
        elif flipping_method == "inpaint_ns":
            for channel in range(datum.shape[2]):
                datum[:, :, channel] = cv2.inpaint(datum[:, :, channel], mask, 3, cv2.INPAINT_NS)
        else:
            raise ValueError("Error in name of distribution to do inpainting. not implemented")

        flipped_data.append(datum)

    return flipped_data


FLIPPING_METHODS = {
    "uniform":  simple_sampling,
    "gaussian": simple_sampling,
    "inpaint_telea": inpainting,
    "inpaint_ns": inpainting,
}

# initialize dataset and dataloader
print(os.path.isdir("../../data/VOC2012"))
# dataset = MyImagenetDataset("../../data/imagenet/imagenet", "val", classidx=["387"])
dataset = VOC2012Dataset("../../data/VOC2012/", "val", classidx=["0"])
dataset.set_mode("preprocessed")

dataloader = DataLoader(dataset, batch_size=2, shuffle=False, startidx=0, endidx=5)

# load example

# load model
model = PytorchModel("../models/pytorch/vgg16_voc/model.pt", "vgg16")
# model = PytorchModel("../models/pytorch/vgg16bn_imagenet/model.pt", "vgg16bn")

# setting = "vgg16bn_imagenet_uncanonized"
setting = "vgg16_voc"

# parameters
layer = "conv1"

R = []

methods = ["IntegratedGradients"]       # , "GradientXActivation"]
# methods = ["alpha2_beta1"]
# methods = ["Saliency", "SmoothGrad", "DeepLift", "GradCam", "IntegratedGradients", "GradientXActivation"]
# methods = ["epsilon", "alpha2_beta1_flat", "epsilon_gamma_box", "epsilon_plus", "epsilon_plus_flat", "epsilon_alpha2_beta1_flat"]
distribution = "inpaint_ns"     # inpaint_ns

percentages = [0.0, 0.002, 0.02, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

# attribute
for xai_method in methods:
    print(xai_method)
    for b, batch in enumerate(dataloader):
        # extract preprocessed data
        data = [sample.datum for sample in batch]
        fnames = [sample.filename.split("/")[-1].split(".")[0] for sample in batch]

        R = model.compute_relevance(data, [layer], 0, xai_method)[layer]

        explanations = np.max(R, axis=3)

        if xai_method in ["Gradient", "SmoothGrad", "LRPZ"]:
            indices = [np.argsort(np.abs(explanation), axis=None) for explanation in explanations]
        else:
            indices = [np.argsort(explanation, axis=None) for explanation in explanations]

        indices = np.array(indices)

        # show pixelflip examples

        for percentage in percentages:

            if percentage == 0:
                flipped_data = np.array([sample.datum for sample in batch])

            else:

                # get first percentage part of pixel indices (lowest relevance)
                # indicesfraction = indices[:, :int(flip_percentage * len(indices))]
                # get last percentage part of pixel indices (highest relevance)
                indicesfraction = indices[:, int((1 - percentage) * indices.shape[1]):]

                flipping_method = FLIPPING_METHODS[distribution]

                flipped_data = flipping_method(batch, indicesfraction, distribution)

            for i, img in enumerate(flipped_data):
                # img = np.load(sample_path + "_" + str(percentage) + ".npy")
                fname = fnames[i]

                # scale to 1
                img += 1.
                img /= 2.

                if np.min(img) < 0.:
                    print(np.min(img))

                if np.max(img) > 1.:
                    print(np.max(img))

                img = img[:, :, ::-1]

                fig = plt.figure()
                plt.axis("off")
                plt.imshow(img)
                # plt.show()
                if setting == "vgg16_voc":
                    plt.savefig("../results/flip_example/{}_{}_{}_{}.png".format(xai_method, fname, distribution, percentage), bbox_inches="tight")
                else:
                    plt.savefig("../results/{}/flip_example/{}_{}_{}_{}.png".format(setting, xai_method, fname, distribution, percentage), bbox_inches="tight")
                plt.close(fig)
