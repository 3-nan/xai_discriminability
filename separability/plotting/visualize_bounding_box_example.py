import os
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import zennit.image

import sys
sys.path.append("/home/motzkus/work/xai_discriminability/separability")
from xaitestframework.dataloading.custom import VOC2012Dataset, MyImagenetDataset
from xaitestframework.dataloading.dataloader import DataLoader
from xaitestframework.models.pytorchmodel import PytorchModel


setting = "vgg16bn_imagenet"
modelname = "vgg16bn"
model_set = "vgg16bn_imagenet"

# initialize dataset and dataloader
print(os.path.isdir("../../data/VOC2012"))
dataset = MyImagenetDataset("../../data/imagenet/imagenet", "val", classidx=["387"])
# dataset = VOC2012Dataset("../../data/VOC2012/", "val", classidx=["3"])
dataset.set_mode("binary_mask")

dataloader = DataLoader(dataset, batch_size=5, shuffle=False, startidx=259, endidx=459)

model = PytorchModel("../models/pytorch/{}/model.pt".format(model_set), modelname)

# xai_method = "epsilon_plus_flat"     # "epsilon_plus"
layer = "conv1"

methods = ["epsilon", "epsilon_plus", "epsilon_plus_flat", "epsilon_gamma_box", "alpha2_beta1", "alpha2_beta1_flat", "epsilon_alpha2_beta1", "epsilon_alpha2_beta1_flat"]
# methods = ["alpha2_beta1", "alpha2_beta1_flat", "epsilon_alpha2_beta1", "epsilon_alpha2_beta1_flat"]
# methods = ["SmoothGrad"]

R = []

# fname = "2008_000082"
# classindices = [3, 13, 14]
#
# image = cv2.imread("../../data/VOC2012/JPEGImages/{}.jpg".format(fname), cv2.IMREAD_COLOR)
# # # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
# image = image.astype(np.float32) / 127.5 - 1.0

##########################################
# compute explanations for chosen sample #
##########################################

# IMAGENET
# fname = "n02509815_1313"        #"n02509815_1425"    # "n01843383_30" "n02165456_1272" "n02509815_1425"
# classindices = [387]
# image = Image.open(os.path.join("../../data/imagenet/imagenet", "bboxes_images", fname.split("_")[0], "{}.JPEG".format(fname)), mode="r").convert(
#             "RGB")
# data_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# image = data_transform(image).permute(1, 2, 0).detach().numpy()
#
# data = [image]
#
# for method in methods:
#
#     for classidx in classindices:
#
#         r = model.compute_relevance(data, [layer], classidx, method)[layer]
#
#         for i, img in enumerate(r):
#             # img = np.abs(img)
#             img_lrp = np.max(img, axis=2)
#             # img_lrp = np.sum(img, axis=2)
#             # img = np.abs(img)
#             # img_lrp = np.linalg.norm(img, axis=2)
#             amax = img_lrp.max((0, 1), keepdims=True)
#             img_lrp = (img_lrp + amax) / 2. / amax
#             # plt.figure()
#             # plt.imshow(img_lrp)
#             # plt.waitforbuttonpress()
#             # plt.close()
#             # zennit.image.imsave("attrs/{}_{}.png".format(xai_method, i), img_lrp, vmin=0., level=2., cmap="bwr")
#             zennit.image.imsave("../results/figures/{}/bbox/{}_{}_{}.png".format(setting, fname, method, classidx),
#                                 img_lrp, vmin=0., vmax=1.,
#                                 level=2., cmap="bwr")


#############################
# here next comment section #
#############################

for b, batch in enumerate(dataloader):

    for sample in batch:

        print(sample.filename)

        # PASCAL VOC
        # image = cv2.imread(sample.filename, cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)

        # IMAGENET
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        im = Image.open(os.path.join("../../data/imagenet/imagenet", "bboxes_images", sample.filename.split("_")[0], sample.filename), mode="r").convert(
            "RGB")

        image_normalized = data_transform(im)

        image = image_normalized.permute(1, 2, 0).detach().numpy()

        # add bounding boxes
        box = sample.binary_mask[list(sample.binary_mask.keys())[0]]
        # box = sample.binary_mask["boat"]
        # print(list(sample.binary_mask.keys())[0])

        box = box[:, :, 0].astype("uint8")
        # box = cv2.flip(box, flipCode=0)
        # box = box.T

        box = np.stack([box, box, box], axis=2)     #.astype(float)

        print(box.shape)

        contours, hierarchy = cv2.findContours(cv2.cvtColor(box, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # apply
        # image = image * box
        image[box == 0] = cv2.blur(image, (5, 5))[box == 0]

        # image = image * 255
        # image = np.float32(image)
        # image = np.array(image)
        # image = image.astype(np.uint8)[:, :, 0]
        # image = cv2.imshow("test", image)
        # cv2.waitKey(0)
        print(np.max(image))
        print("{} contour(s) found".format(len(contours)))
        print(image.shape)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.drawContours(image, contours, -1, (0, 0, 255), 1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # image = image + 0.2 * box

        # show
        plt.figure()
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.margins(0.0)
        plt.imshow(image)
        # plt.waitforbuttonpress()
        # plt.close()

        plt.savefig("../results/figures/bbox_imagenet/{}.png".format(sample.filename.split("/")[-1].split(".")[0]), format="png", bbox_inches="tight")
        plt.close()

####################################################
# single imagenet sample drawn with bounding box   #
####################################################

