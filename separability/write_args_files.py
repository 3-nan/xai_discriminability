from xaitestframework.dataloading import Dataloader

partition = "train" # "val" "test"
rule = "LRPSequentialCompositeA"
# layers = ['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_4', 'conv2d_7', 'conv2d_10', 'dense', 'dense_1', 'dense_2']
layers = ['input_1', 'block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1', 'fc1', 'fc2']

dp = Dataloader("../../data/imagenet/", 50)
dataset = dp.get_data(partition)
cardinality = dataset.cardinality().numpy()
print(cardinality)

job_size = 1000     # number of images to process per job
argument_list = []

for layer in layers:

    for c in [96, 292, 301, 347, 387, 417, 604, 890, 937, 954]: # range(10):
        base_args = ""

        base_args += "-d /data/cluster/users/motzkus/data/imagenet/"
        base_args += " -dn imagenet"
        base_args += " -o /data/cluster/users/motzkus/relevance_maps"
        base_args += " -m /data/cluster/users/motzkus/models/untrained_vgg16_imagenet/model/"
        base_args += " -mn vgg16"

        for i in range(int(cardinality*50/job_size)):        # 50000/job_size)):

            args = base_args + " -si " + str(i*job_size) + " -ei " + str((i+1)*job_size)
            args += " -p " + partition
            args = args + " -cl " + str(c)
            args = args + " -r " + rule
            args = args + " -l " + layer
            args += " -bs 50"

            argument_list.append(args)

# write out to file
output_file = "test_pathname" + '.args'
print('writing converted arguments to {}'.format(output_file))
with open(output_file, 'wt') as f:
    f.write('\n'.join(argument_list))
