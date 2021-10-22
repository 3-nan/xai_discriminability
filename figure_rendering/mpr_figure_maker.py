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
setting = "cascading_bottom_up"     # "cascading_top_down"      "independent"   cascading_bottom_up
template = "results/inkscape/mpr_{}.svg".format(setting)

fname = "2008_000804"       # "n01843383_12472"

layers = ["linear3", "linear2", "linear1", "conv13", "conv11", "conv8", "conv5", "conv3", "conv1"]

if setting != "cascading_top_down":
    layers = layers[::-1]
xai_methods = ["Saliency", "SmoothGrad", "IntegratedGradients", "GradientXActivation", "GradCam", "DeepLift",
               "epsilon", "alpha2_beta1_flat", "epsilon_gamma_box", "epsilon_plus", "epsilon_plus_flat",
               "epsilon_alpha2_beta1_flat"]

output_path = "results/figures/mpr/{}_{}.svg".format(setting, fname)
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
newpath = "../data/VOC2012/JPEGImages/{}.jpg".format(fname)
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

    # set attribution
    newpath = "results/model_parameter_randomization/figure_examples/{}/{}_original.png".format(xai_method, fname)
    abspath = os.path.abspath(newpath)
    if not os.path.isfile(abspath):
        print("no file for method {}".format(xai_method))
    elem = tree.findall('''.//*[@id='image-{}-original']'''.format(x))[0]
    elem.set(href, abspath)
    elem.set(absref, abspath)

    for l, layer in enumerate(layers):

        # set randomized attribution
        newpath = "results/model_parameter_randomization/figure_examples/{}/{}_{}_{}.png".format(xai_method, fname, setting, layer)
        abspath = os.path.abspath(newpath)
        if not os.path.isfile(abspath):
            print("no file")

        elem = tree.findall('''.//*[@id='image-{}-{}']'''.format(x, l))[0]
        elem.set(href, abspath)
        elem.set(absref, abspath)

tree.write(output_path)
