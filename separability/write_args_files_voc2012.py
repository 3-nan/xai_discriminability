import math
from xaitestframework.dataloading.custom import get_dataset

data_path = "/data/cluster/users/motzkus/data/VOC2012/"
data_name = "voc2012"
dataloader_name = "VOC2012Dataloader" # not used anymore??
dataset_name = "VOC2012Dataset"
classindices = range(20)

output_dir = "/data/cluster/users/motzkus/relevance_maps"

model_path = "/data/cluster/users/motzkus/models/vgg16_voc2012/model"
model_name = "vgg16"

partition = "train"  # "train" "val" "test"

rules = ["LRPSequentialCompositeA", "LRPSequentialCompositeBFlat", "LRPZ", "LRPGamma", "LRPAlpha1Beta0", "Gradient", "SmoothGrad"]

# layers = ['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_4', 'conv2d_7', 'conv2d_10', 'dense', 'dense_1', 'dense_2']
layers = ['input_1', 'block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1', 'fc1', 'fc2']
batch_size = 50

dataset = get_dataset(dataset_name)
dataset = dataset("../../data/VOC2012/", partition)

print(len(dataset))

job_size = 1000     # number of images to process per job
argument_list = []

for rule in rules:

    for c in classindices:
        base_args = ""

        base_args += "-d " + data_path
        base_args += " -dn " + data_name
        base_args += " -dl " + dataset_name
        base_args += " -o " + output_dir
        base_args += " -m " + model_path
        base_args += " -mn " + model_name

        for i in range(math.ceil(len(dataset)/job_size)):        # 50000/job_size)):

            args = base_args + " -si " + str(i*job_size) + " -ei " + str((i+1)*job_size)
            args += " -p " + partition
            args = args + " -cl " + str(c)
            args = args + " -r " + rule
            args = args + " -l " + ":".join(layers)
            args += " -bs " + str(batch_size)

            argument_list.append(args)

# write out to file
output_file = "relevance" + ".args"
print('writing converted arguments to {}'.format(output_file))
with open(output_file, 'wt') as f:
    f.write('\n'.join(argument_list))
