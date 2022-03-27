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
# setting = "VOC2012_vgg16"     # VOC2012_vgg16
# setting = "imagenet_vgg16bn_uncanonized"
setting = "imagenet_resnet18_uncanonized"
# template = "results/inkscape/separability_example.svg"
template = "results/inkscape/separability_example_imagenet.svg"

# fname = "2008_000133"       # "n01843383_12472"     2008_000090
fname = "n01843383_4067"

# layer = "linear1"
layer = "conv19"
# layer = "conv1"

# classindices = [0, 2, 3, 4]
# true_idx = 1
classindices = [126, 155, 292, 301]
true_idx = 96

# xai_methods = ["IntegratedGradients", "GradientXActivation", "GradCam", "DeepLift",
#                "epsilon", "alpha2_beta1", "epsilon_gamma_box", "epsilon_plus", "epsilon_plus_flat",
#                "epsilon_alpha2_beta1_flat"]
xai_methods = ["epsilon", "alpha2_beta1", "epsilon_gamma_box", "epsilon_plus", "epsilon_alpha2_beta1"]

output_path = "results/figures/separability/{}_{}_{}.svg".format(setting, fname, layer)
print(output_path)

tree = ET.parse(template)
print(tree)
print(list(tree.iter()))


href = '{%s}href' % ns['xlink']
absref = '{%s}abshref' % ns['sodipodi']

# set input image
# newpath = "../data/VOC2012/JPEGImages/{}.jpg".format(fname)
newpath = "../data/imagenet/imagenet/bboxes_images/{}/{}.JPEG".format(fname.split("_")[0], fname)
abspath = os.path.abspath(newpath)
if not os.path.isfile(abspath):
    print("no file")
elem = tree.findall('''.//*[@id='image-original']''')[0]
elem.set(href, abspath)
elem.set(absref, abspath)
print("original")

# set attribution
# newpath = "results/attributions/{}.png".format(sample)
# abspath = os.path.abspath(newpath)
# if not os.path.isfile(abspath):
#     print("no file")
# elem = tree.findall('''.//*[@id='image-attribution']''')[0]
# elem.set(href, abspath)
# elem.set(absref, abspath)

# set auc
# newpath = "results/figures/pixelflipping/auc.svg"
# abspath = os.path.abspath(newpath)
# if not os.path.isfile(abspath):
#     print("no file")
# elem = tree.findall('''.//*[@id='image-auc']''')[0]
# elem.set(href, abspath)
# elem.set(absref, abspath)

for x, xai_method in enumerate(xai_methods):

    print(xai_method)
    # set attribution
    newpath = "results/separability/figures/{}/feature_maps/{}_{}_{}_{}.png".format(setting, fname, layer, xai_method, true_idx)
    abspath = os.path.abspath(newpath)
    if not os.path.isfile(abspath):
        print("no file for method {}".format(xai_method))
    elem = tree.findall('''.//*[@id='image-{}-original']'''.format(x))[0]
    elem.set(href, abspath)
    elem.set(absref, abspath)

    for i, idx in enumerate(classindices):

        # set randomized attribution
        newpath = "results/separability/figures/{}/feature_maps/{}_{}_{}_{}.png".format(setting, fname, layer, xai_method, idx)
        abspath = os.path.abspath(newpath)
        if not os.path.isfile(abspath):
            print("no file")

        elem = tree.findall('''.//*[@id='image-{}-{}']'''.format(x, i))[0]
        elem.set(href, abspath)
        elem.set(absref, abspath)

tree.write(output_path)
