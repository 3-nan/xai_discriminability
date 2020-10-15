import os

job_size = 1000     # number of images to process per job
argument_list = []

for c in range(10):
    base_args = ""

    base_args += "-dn cifar10"
    base_args += " -o /data/cluster/users/motzkus/relevance_maps"
    base_args += " -m /data/cluster/users/motzkus/models/vgg16_cifar10/model/"
    base_args += " -mn vgg16"

    for i in range(int(50000/job_size)):

        args = base_args + " -si " + str(i*job_size) + " -ei " + str((i+1)*job_size)
        args += " -p train"
        args = args + " -cl " + str(c)
        args += " -r LRPZ"
        args += " -l conv2d"
        args += " -bs 50"

        argument_list.append(args)

# write out to file
output_file = "test_pathname" + '.args'
print('writing converted arguments to {}'.format(output_file))
with open(output_file, 'wt') as f:
    f.write('\n'.join(argument_list))