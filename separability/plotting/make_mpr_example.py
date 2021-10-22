# Compute MPR example for single samples
import os
import sys
import numpy as np
from zennit.image import imsave

sys.path.append("/home/motzkus/work/xai_discriminability/separability")
from xaitestframework.dataloading.custom import VOC2012Dataset, MyImagenetDataset
from xaitestframework.dataloading.dataloader import DataLoader
from xaitestframework.models.pytorchmodel import PytorchModel
from xaitestframework.experiments.model_parameter_randomization import spearman_distance


def plot_and_save_explanation(path, img):

    img_lrp = np.sum(img, axis=2)
    amax = img_lrp.max((0, 1), keepdims=True)
    img_lrp = (img_lrp + amax) / 2. / amax
    imsave(path, img_lrp, vmin=0., vmax=1., level=2., cmap="bwr")

# init dataset and dataloader
classidx = 96
data_path = "../../data/imagenet/imagenet"
dataset = MyImagenetDataset(data_path, "val", classidx=[classidx])
dataset.set_mode("preprocessed")

dataloader = DataLoader(dataset, batch_size=5, shuffle=False, startidx=0, endidx=5)
print("Data loaded.")

# init model
model_path = "../models/pytorch/vgg16bn_imagenet/model.pt"
model_name = "vgg16bn"
model = PytorchModel(model_path, model_name)
print("Model loaded.")

# parameters
result_dir = "../results/vgg16bn_imagenet/mpr_example"
xai_methods = ["epsilon", "alpha2_beta1", "alpha2_beta1_flat",
               "epsilon_plus", "epsilon_plus_flat", "epsilon_gamma_box", "epsilon_alpha2_beta1",
               "epsilon_alpha2_beta1_flat"]

setting = "cascading_top_down"    # cascading_top_down  independent     cascading_bottom_up
distance = "spearman_distance"
input_layer = "conv1"
canonization = "canonized"

# get layers including weights and iterate them
layer_names = model.get_layer_names(with_weights_only=True)

if setting == "cascading_top_down":
    layer_names = layer_names[::-1]

# compute original explanations
for b, batch in enumerate(dataloader):
    data = [sample.datum for sample in batch]
    fnames = [sample.filename.split(".")[0] for sample in batch]


for xai_method in xai_methods:
    print(xai_method)

    original_explanations = model.compute_relevance(data, [input_layer], neuron_selection=int(classidx),
                                                    xai_method=xai_method, additional_parameter=None)

    original_explanations = original_explanations[input_layer]
    print("original explanations shape: {}".format(original_explanations.shape))

    for i, explanation in enumerate(original_explanations):
        plot_and_save_explanation(
            os.path.join(result_dir, setting, "{}_{}_{}.png".format(fnames[i], xai_method, canonization)),
            explanation)

for layer_name in layer_names:

    print(layer_name)

    # randomize layer weights
    if setting == "independent":
        model = PytorchModel(model_path, model_name)

    model = model.randomize_layer_weights(layer_name)

    for xai_method in xai_methods:

        explanations = model.compute_relevance(data, [input_layer], neuron_selection=int(classidx),
                                               xai_method=xai_method, additional_parameter=None)

        for i, explanation in enumerate(explanations[input_layer]):

            plot_and_save_explanation(os.path.join(result_dir, setting, "{}_{}_{}_{}.png".format(fnames[i], xai_method,
                                                                                                 layer_name,
                                                                                                 canonization)),
                                      explanation)
