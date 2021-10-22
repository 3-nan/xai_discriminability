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
setting = "inpaint_ns"     # "cascading_top_down"      "independent"
template = "results/inkscape/flip_example.svg".format(setting)
fname = "2008_000804"

percentages = [0.002, 0.02, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

xai_methods = ["Saliency", "SmoothGrad", "IntegratedGradients", "GradientXActivation", "GradCam", "DeepLift", "epsilon", "alpha2_beta1_flat", "epsilon_gamma_box", "epsilon_plus", "epsilon_plus_flat", "epsilon_alpha2_beta1_flat"]

output_path = "results/figures/flip_example/{}_{}.svg".format(setting, fname)
print(output_path)

tree = ET.parse(template)
print(tree)
print(list(tree.iter()))


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

for x, xai_method in enumerate(xai_methods):

    # set attribution
    newpath = "results/model_parameter_randomization/figure_examples/{}/{}_original.png".format(xai_method, fname)
    abspath = os.path.abspath(newpath)
    if not os.path.isfile(abspath):
        print("no file for method {}".format(xai_method))
    elem = tree.findall('''.//*[@id='image-{}-original']'''.format(x))[0]
    elem.set(href, abspath)
    elem.set(absref, abspath)

    for p, percentage in enumerate(percentages):

        # set randomized attribution
        newpath = "results/flip_example/{}_{}_{}_{}.png".format(xai_method, fname, setting, percentage)
        abspath = os.path.abspath(newpath)
        if not os.path.isfile(abspath):
            print("no file")

        elem = tree.findall('''.//*[@id='image-{}-{}']'''.format(x, p))[0]
        elem.set(href, abspath)
        elem.set(absref, abspath)

tree.write(output_path)
