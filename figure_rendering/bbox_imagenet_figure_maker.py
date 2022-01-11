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
template = "results/inkscape/bbox_imagenet_example_extended.svg"
# fname = "n02509815_1313"
# fname = "n01843383_4067"
fname = "n01843383_10599"

# setting = "vgg16bn_imagenet"
setting = "resnet18_imagenet_uncanonized"
# n02509815_1313_alpha2_beta1_387.png

# classidx = "387"

# "Saliency", "SmoothGrad", "IntegratedGradients", "GradientXActivation", "GradCam", "DeepLift",
# xai_methods = ["epsilon", "alpha2_beta1_flat", "epsilon_gamma_box", "epsilon_plus", "epsilon_plus_flat", "epsilon_alpha2_beta1_flat"]
xai_methods = ["epsilon", "alpha2_beta1", "alpha2_beta1_flat", "epsilon_gamma_box",
               "epsilon_plus", "epsilon_plus_flat", "epsilon_alpha2_beta1", "epsilon_alpha2_beta1_flat"]


# output_path = "results/figures/bbox_imagenet/example2_{}_{}.svg".format(setting, fname)
output_path = "results/figures/{}/bbox/{}.svg".format(setting, fname)
print(output_path)

tree = ET.parse(template)
print(tree)
print(list(tree.iter()))
# for elem in tree.iterfind('.//text', ns):
#     tspan = next(iter(elem))
#     newtext = "new_text"
#     tspan.text = newtext
#
# for elem in tree.iterfind('.//path', ns):
#     path = elem
#     newstyle = "new_style"
#
# for elem in tree.iterfind('.//image', ns):
#     print("elem found")
#     href = '{%s}href' % ns['xlink']
#     absref = '{%s}abshref' % ns['sodipodi']
#     eid = elem.get('id')
#     print(eid)
#
#     oldpath = elem.get(href, '')
#     newpath = "results/figures/manifold_pixelflipping_input.svg"
#
#     if not newpath:
#         logger.warning("Empty path for node \'%s\'", eid)
#         continue
#
#     abspath = os.path.abspath(newpath)
#     if not os.path.isfile(abspath):
#         logger.warning("File not found for node \'%s\': \'%s\'", eid, abspath)
#         continue
#
#     elem.set(href, abspath)
#     elem.set(absref, abspath)


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

for x, xai_method in enumerate(xai_methods):

    # set attribution
    # newpath = "results/figures/{}/bbox/{}_{}_{}.png".format(setting, fname, xai_method, classidx)
    newpath = "results/separability/figures/imagenet_resnet18_uncanonized/feature_maps/{}_{}_{}_{}.png".format(fname, "conv1", xai_method, 96)
    abspath = os.path.abspath(newpath)
    if not os.path.isfile(abspath):
        print("no file for method {}".format(xai_method))
    elem = tree.findall('''.//*[@id='image-{}']'''.format(x))[0]
    elem.set(href, abspath)
    elem.set(absref, abspath)

tree.write(output_path)
