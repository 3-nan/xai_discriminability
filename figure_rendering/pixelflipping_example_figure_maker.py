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
template = "results/inkscape/pixelflipping_examples_space_final.svg"
sample = "2008_000804"

distributions = ["uniform", "gaussian", "inpaint_telea", "inpaint_ns"]

percentages = [0.0002, 0.002, 0.02, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

output_path = "results/figures/pixelflipping/{}_final.svg".format(sample)
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

# set original
newpath = "results/pixelflipping_example/{}_{}_{}.svg".format(sample, "uniform", 0.0)
abspath = os.path.abspath(newpath)
if not os.path.isfile(abspath):
    print("no file")
elem = tree.findall('''.//*[@id='image-original']''')[0]
elem.set(href, abspath)
elem.set(absref, abspath)
print("original")

# set attribution
newpath = "results/attributions/{}.png".format(sample)
abspath = os.path.abspath(newpath)
if not os.path.isfile(abspath):
    print("no file")
elem = tree.findall('''.//*[@id='image-attribution']''')[0]
elem.set(href, abspath)
elem.set(absref, abspath)

# set auc
newpath = "results/figures/pixelflipping/auc.svg"
abspath = os.path.abspath(newpath)
if not os.path.isfile(abspath):
    print("no file")
elem = tree.findall('''.//*[@id='image-auc']''')[0]
elem.set(href, abspath)
elem.set(absref, abspath)

for d, distribution in enumerate(distributions):
    for p, percentage in enumerate(percentages):

        newpath = "results/pixelflipping_example/{}_{}_{}.svg".format(sample, distribution, percentage)
        abspath = os.path.abspath(newpath)
        if not os.path.isfile(abspath):
            print("no file")

        elem = tree.findall('''.//*[@id='image-{}-{}']'''.format(d, p))[0]
        elem.set(href, abspath)
        elem.set(absref, abspath)

tree.write(output_path)
