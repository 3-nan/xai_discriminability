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
# setting = "cascading_bottom_up"     # "cascading_top_down"      "independent"
# template = "results/inkscape/bbox_imagenet_example_2.svg"
template = "results/inkscape/canonization_example_final.svg"
# fname = "n02509815_1313"
# fname = "n01843383_4067"
# fname = "n01843383_10599"          # n01843383_10599    n01843383_12472      n01843383_4067
# classidx = 96

fname= "n02509815_14698"         # n02509815_3173
classidx = 387

modelname = "vgg16bn"      # resnet18  vgg16bn

setting = "imagenet_{}".format(modelname)
# setting = "imagenet_resnet18"
# n02509815_1313_alpha2_beta1_387.png

# "Saliency", "SmoothGrad", "IntegratedGradients", "GradientXActivation", "GradCam", "DeepLift",
# xai_methods = ["epsilon", "alpha2_beta1_flat", "epsilon_gamma_box", "epsilon_plus", "epsilon_plus_flat", "epsilon_alpha2_beta1_flat"]
# xai_methods = ["epsilon", "epsilon_plus", "epsilon_alpha2_beta1", "guided_backprop", "excitation_backprop", "Gradient", "IntegratedGradients"]
xai_methods = ["epsilon", "epsilon_plus", "epsilon_plus_flat", "epsilon_alpha2_beta1", "epsilon_alpha2_beta1_flat", "excitation_backprop", "guided_backprop", "Gradient", "IntegratedGradients", "GradientxInput"]


# output_path = "results/figures/bbox_imagenet/example2_{}_{}.svg".format(setting, fname)
output_path = "results/figures/canonization/{}/{}_{}_final.svg".format(setting, fname, modelname)
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
    print("no file")
elem = tree.findall('''.//*[@id='image-original-0']''')[0]
elem.set(href, abspath)
elem.set(absref, abspath)

print("original done")

for s, state in enumerate(["canonized", "noncanonized"]):
    for x, xai_method in enumerate(xai_methods):

        # set attribution
        # newpath = "results/figures/{}/bbox/{}_{}_{}.png".format(setting, fname, xai_method, classidx)
        if state == "canonized":
            newpath = "results/attributions_canonization/{}/{}/{}_{}.png".format(setting, xai_method, classidx, fname)
        else:
            newpath = "results/attributions_canonization/{}_noncanonized/{}/{}_{}.png".format(setting, xai_method, classidx, fname)
        abspath = os.path.abspath(newpath)
        if not os.path.isfile(abspath):
            print("no file for method {}".format(xai_method))
        elem = tree.findall('''.//*[@id='image-{}-{}']'''.format(s, x))[0]
        elem.set(href, abspath)
        elem.set(absref, abspath)

tree.write(output_path)
