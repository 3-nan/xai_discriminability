from xaitestframework.dataloading.dataloader import get_dataloader

data_path = "/data/cluster/users/motzkus/data/imagenet/"
data_name = "imagenet"
dataloader_name = "ImagenetDataloader"

output_dir = "/data/cluster/users/motzkus/relevance_maps"

model_path = "/data/cluster/users/motzkus/models/vgg16_imagenet/model"
model_name = "vgg16"

partition = "train" # "val" "test"

rules = ["LRPSequentialCompositeA", "LRPSequentialCompositeBFlat", "LRPZ", "LRPGamma", "LRPAlpha1Beta0", "Gradient", "SmoothGrad"]

# layers = ['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_4', 'conv2d_7', 'conv2d_10', 'dense', 'dense_1', 'dense_2']
layers = ['input_1', 'block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1', 'fc1', 'fc2']

dataloader = get_dataloader(dataloader_name)
dataloader = dataloader("../../data/imagenet/", partition, 50)

print(len(dataloader))

job_size = 1000     # number of images to process per job
argument_list = []

for rule in rules:

    for c in [96, 292, 301, 347, 387, 417, 604, 890, 937, 954]: # range(10):
        base_args = ""

        base_args += "-d " + data_path
        base_args += " -dn " + data_name
        base_args += " -dl " + dataloader_name
        base_args += " -o " + output_dir
        base_args += " -m " + model_path
        base_args += " -mn " + model_name

        for i in range(int(len(dataloader)/job_size)):        # 50000/job_size)):

            args = base_args + " -si " + str(i*job_size) + " -ei " + str((i+1)*job_size)
            args += " -p " + partition
            args = args + " -cl " + str(c)
            args = args + " -r " + rule
            args = args + " -l " + ":".join(layers)
            args += " -bs 50"

            argument_list.append(args)

# write out to file
output_file = "relevance" + ".args"
print('writing converted arguments to {}'.format(output_file))
with open(output_file, 'wt') as f:
    f.write('\n'.join(argument_list))
