# This is an example of using the xaitestframework
import os
from xaitestframework.dataloading.dataloader import DataLoader
from xaitestframework.dataloading.custom import get_dataset
from xaitestframework.helpers.model_helper import init_model
from xaitestframework.helpers.universal_helper import compute_relevance_path
from xaitestframework.experiments.attribution_computation import combine_path, compute_attributions_for_class
from xaitestframework.experiments.pixelflipping import compute_pixelflipping_score

#######################################
# example for explanation computation #
#######################################

# set parameters
batch_size = 50
layer_names = ["input_1", "block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1", "fc1", "fc2"]
xai_method = "LRPSequentialCompositeA"
classidx = 7

# init model
model = init_model("model_dir/my_model", "my_modelname", framework="myframework")

# initialize dataset
dataset = get_dataset("voc_2012")
dataset = dataset("datasets/my_dataset", "train")

# compute/create output dir
output_dir = combine_path("explanations/", ["voc_2012", "vgg16"])

compute_attributions_for_class(dataset, "train", batch_size, model, layer_names,
                               xai_method, str(classidx), output_dir, startidx=0, endidx=200)

#############################
# example for pixelflipping #
#############################

# set parameters
batch_size = 50
layer_names = ["input_1", "block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1", "fc1", "fc2"]
xai_method = "LRPSequentialCompositeA"
classidx = 7
distribution = "gaussian"
percentage_values = [0.1, 0.2, 0.3, 0.4, 0.5]

# init model
model = init_model("model_dir/my_model", "my_modelname", framework="myframework")

# initialize dataset
dataset = get_dataset("voc_2012")
dataset = dataset("datasets/my_dataset", "val")

dataloader = DataLoader(dataset, batch_size=batch_size)

# compute/create output dir
explanation_dir = compute_relevance_path("explanations/", "voc_2012", "vgg16", layer_names[0], xai_method)
explanation_dir = os.path.join(explanation_dir, "val")

scores = compute_pixelflipping_score(dataloader, model, explanation_dir, classidx, xai_method, distribution,
                                     percentage_values)

for p in percentage_values:
    print("Score for {} of pixels flipped: {}".format(p, scores[p]))
