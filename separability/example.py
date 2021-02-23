# This is an example of using the xaitestframework
from xaitestframework.dataloading.custom import get_dataset
from xaitestframework.helpers.model_helper import init_model
from xaitestframework.experiments.attribution_computation import combine_path, compute_explanations_for_class

#######################################
# example for explanation computation #
#######################################

# set parameters
batch_size = 50
layer_names = ["input_1", "block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1", "fc1", "fc2"]
xai_method = "LRPSequentialCompositeA"
classidx = 7

# init model
model = init_model("model_dir/my_model")

# initialize dataset
dataset = get_dataset("voc_2012")
dataset = dataset("datasets/my_dataset", "train")

# compute/create output dir
output_dir = combine_path("explanations/", ["voc_2012", "vgg16"])

compute_explanations_for_class(dataset, "train", batch_size, model, layer_names,
                               xai_method, str(classidx), output_dir, startidx=0, endidx=200)

#############################
# example for pixelflipping #
#############################
