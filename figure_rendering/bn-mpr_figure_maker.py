import xml.etree.ElementTree as ET
import logging
import os


logger = logging.getLogger(__name__)

ns = {
    "svg": "http://www.w3.org/2000/svg",
    "": "http://www.w3.org/2000/svg",
    "xlink": "http://www.w3.org/1999/xlink",
    "sodipodi": "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"
}

# parameters
setting = "cascading_top_down"     # "cascading_top_down"      "independent"   cascading_bottom_up
canonization = "uncanonized"
# template = "results/inkscape/mpr_{}_bn.svg".format(setting)
template = "results/inkscape/mpr_{}_resnet.svg".format(setting)

# modelsetting = "vgg16bn_imagenet"
modelsetting = "resnet18_imagenet"
fname = "n01843383_4067"       # "n01843383_12472"     n01843383_10599     n01843383_4067  2008_000804

# layers = ["linear3", "linear2", "linear1", "conv13", "conv11", "conv8", "conv5", "conv3", "conv1"]
layers = ["linear1", "conv19", "conv16", "conv14", "conv11", "conv9", "conv6", "conv4", "conv1"]


if setting != "cascading_top_down":
    layers = layers[::-1]
xai_methods = ["epsilon", "alpha2_beta1", "alpha2_beta1_flat", "epsilon_gamma_box", "epsilon_plus", "epsilon_plus_flat",
               "epsilon_alpha2_beta1", "epsilon_alpha2_beta1_flat"]

#   "Saliency", "SmoothGrad", "IntegratedGradients", "GradientXActivation", "GradCam", "DeepLift",

output_path = "results/figures/{}/mpr/{}_{}_{}.svg".format(modelsetting, setting, fname, canonization)
print(output_path)

tree = ET.parse(template)
print(tree)
print(list(tree.iter()))

href = '{%s}href' % ns['xlink']
absref = '{%s}abshref' % ns['sodipodi']

# set input image
newpath = "../data/imagenet/imagenet/bboxes_images/{}/{}.JPEG".format(fname.split("_")[0], fname)
abspath = os.path.abspath(newpath)
if not os.path.isfile(abspath):
    print("no file {}".format(abspath))
    print(os.path.isdir("../data/imagenet/imagenet/bboxes_images"))
elem = tree.findall('''.//*[@id='image-original']''')[0]
elem.set(href, abspath)
elem.set(absref, abspath)
print("original")


for x, xai_method in enumerate(xai_methods):

    # set attribution
    newpath = "results/{}/mpr_example/{}/{}_{}_{}.png".format(modelsetting, setting, fname, xai_method, canonization)
    abspath = os.path.abspath(newpath)
    if not os.path.isfile(abspath):
        print("no file for method {}".format(xai_method))
    elem = tree.findall('''.//*[@id='image-{}-original']'''.format(x))[0]
    elem.set(href, abspath)
    elem.set(absref, abspath)

    for l, layer in enumerate(layers):

        # set randomized attribution
        newpath = "results/{}/mpr_example/{}/{}_{}_{}_{}.png".format(modelsetting, setting, fname, xai_method, layer, canonization)
        abspath = os.path.abspath(newpath)
        if not os.path.isfile(abspath):
            print("no file")

        elem = tree.findall('''.//*[@id='image-{}-{}']'''.format(x, l))[0]
        elem.set(href, abspath)
        elem.set(absref, abspath)

tree.write(output_path)
