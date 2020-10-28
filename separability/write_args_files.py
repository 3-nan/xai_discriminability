import os

rule = "LRPSequentialCompositeA"
layers = ['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_4', 'conv2d_7', 'conv2d_10', 'dense', 'dense_1', 'dense_2']

job_size = 1000     # number of images to process per job
argument_list = []

for layer in layers:

    for c in range(10):
        base_args = ""

        base_args += "-dn cifar10"
        base_args += " -o /data/cluster/users/motzkus/relevance_maps"
        base_args += " -m /data/cluster/users/motzkus/models/vgg16_cifar10/model/"
        base_args += " -mn vgg16"

        for i in range(int(10000/job_size)):

            args = base_args + " -si " + str(i*job_size) + " -ei " + str((i+1)*job_size)
            args += " -p test"
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
