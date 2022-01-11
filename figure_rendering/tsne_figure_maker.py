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

multi_class = True


if multi_class == True:
    # parameters
    template = "results/inkscape/pixelflipping_tsne_new.svg"

    tree = ET.parse(template)
    print(tree)
    print(list(tree.iter()))

    layer = "conv13"           # conv13    linear2
    distributions = ["uniform", "gaussian", "inpaint_telea", "inpaint_ns"]

    for distribution in distributions:

        classindices = [3, 8, 12]
        # percentages = [0.0, 0.0002, 0.002, 0.02, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

        percentages = [0.0, 0.02, 0.2, 0.5, 0.9]

        output_path = "results/figures/tsne_manifold/tsne_{}_{}_new.svg".format(layer, distribution)

        for p, percentage in enumerate(percentages):
            for i, idx in enumerate(classindices):
                href = '{%s}href' % ns['xlink']
                absref = '{%s}abshref' % ns['sodipodi']

                newpath = "results/figures/tsne_manifold/{}/{}_{}_{}.svg".format(layer, distribution, idx, percentage)
                abspath = os.path.abspath(newpath)

                elem = tree.findall('''.//*[@id='image{}-{}']'''.format(i+1, p+1))[0]
                elem.set(href, abspath)
                elem.set(absref, abspath)

        tree.write(output_path)

else:
    template = "results/inkscape/tsne_one_class.svg"

    tree = ET.parse(template)
    print(tree)
    print(list(tree.iter()))

    layer = "linear2"
    distributions = ["uniform", "gaussian", "inpaint_telea", "inpaint_ns"]

    output_path = "results/figures/tsne_manifold_one_class/tsne_single_{}.svg".format(layer)

    for distribution in distributions:

        href = '{%s}href' % ns['xlink']
        absref = '{%s}abshref' % ns['sodipodi']

        newpath = "results/figures/tsne_manifold_one_class/{}_{}.svg".format(layer, distribution)
        abspath = os.path.abspath(newpath)

        elem = tree.findall('''.//*[@id='image-{}']'''.format(distribution))[0]
        elem.set(href, abspath)
        elem.set(absref, abspath)

    tree.write(output_path)
