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
template = "results/inkscape/bbox_example.svg"
fname = "2008_000082"

classindices = ["3", "13", "14"]

xai_methods = ["Saliency", "SmoothGrad", "IntegratedGradients", "GradientXActivation", "GradCam", "DeepLift", "epsilon", "alpha2_beta1_flat", "epsilon_gamma_box", "epsilon_plus", "epsilon_plus_flat", "epsilon_alpha2_beta1_flat"]

output_path = "results/figures/bbox/example_{}.svg".format(fname)
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
newpath = "results/figures/bbox/{}_boat.png".format(fname)
abspath = os.path.abspath(newpath)
if not os.path.isfile(abspath):
    print("no file")
elem = tree.findall('''.//*[@id='image-original-0']''')[0]
elem.set(href, abspath)
elem.set(absref, abspath)

newpath = "results/figures/bbox/{}_motorbike.png".format(fname)
abspath = os.path.abspath(newpath)
if not os.path.isfile(abspath):
    print("no file")
elem = tree.findall('''.//*[@id='image-original-1']''')[0]
elem.set(href, abspath)
elem.set(absref, abspath)

newpath = "results/figures/bbox/{}_person.png".format(fname)
abspath = os.path.abspath(newpath)
if not os.path.isfile(abspath):
    print("no file")
elem = tree.findall('''.//*[@id='image-original-2']''')[0]
elem.set(href, abspath)
elem.set(absref, abspath)
print("original")

for x, xai_method in enumerate(xai_methods):

    for c, classidx in enumerate(classindices):

        # set attribution
        newpath = "results/figures/bbox/{}_{}_{}.png".format(fname, xai_method, classidx)
        abspath = os.path.abspath(newpath)
        if not os.path.isfile(abspath):
            print("no file for method {}".format(xai_method))
        elem = tree.findall('''.//*[@id='image-{}-{}']'''.format(x, c))[0]
        elem.set(href, abspath)
        elem.set(absref, abspath)

tree.write(output_path)
