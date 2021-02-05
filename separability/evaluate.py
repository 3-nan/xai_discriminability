import os
import sys
import math
import yaml
import subprocess

from xaitestframework.experiments.relevance_computation import compute_relevances_for_class
from xaitestframework.dataloading.custom import get_dataset
from xaitestframework.helpers.universal_helper import join_path


def submit_on_sungrid(args, configs, jobconfig, quantification_method, index):
    """ Submit the specified job on a sungrid engine. """

    print('Generating and submitting jobs for:')
    print('    {}'.format(args))

    # generate script
    execthis = ['#!/bin/bash']
    execthis = ['source /home/fe/motzkus/.bashrc']  # enables conda for bash ToDo: change dir
    # execthis += ['cd {}/experiments'.format(HERE)]  # go to python root
    execthis += ['{} activate {}'.format(configs['system_config']['conda'], configs['system_config']['environment'])]  # enter venv
    execthis += ['python3 -m xaitestframework.experiments.{} {}'.format(quantification_method, args)]  # call script with parameters.
    execthis += ['{} deactivate'.format(configs['system_config']['conda'])]  # leave venv
    execthis = '\n'.join(execthis)

    JOBNAME = '[{}_of_{}]_of_"{}"'.format(index + 1, "X", quantification_method)
    SCRIPTFILE = '{}/{}-{}'.format(configs['dirs']['scriptdir'], quantification_method, index + 1)
    print('    as file: {}'.format(SCRIPTFILE))
    print('    as name: {}'.format(JOBNAME))

    # write to script
    with open(SCRIPTFILE, 'w') as f:
        f.write(execthis)
        print('    job written to {}'.format(SCRIPTFILE))
        os.system('chmod 777 {}'.format(SCRIPTFILE))

    # job submission
    cmd = ['qsub']
    cmd += ['-N', JOBNAME]
    cmd += ['-pe', 'multi', str(configs['system_config']['threads'])]
    cmd += ['-e', configs['dirs']['logdir']]
    cmd += ['-o', configs['dirs']['logdir']]
    cmd += ['-l', 'h_vmem=' + jobconfig['memory_per_job']]
    cmd += [SCRIPTFILE]
    print('    preparing to submit via command: {}'.format(cmd))

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = p.communicate()
    print('    ', 'P:', p, 'O:', stdout, 'E:', stderr)
    if (p.returncode == 0):
        id_job = stdout.decode('utf-8').split()[2]
    else:
        id_job = None

    if len(stderr) > 0:
        print('ERROR during job submission? "{}"'.format(stderr))
        exit()

    print('    dispatched job {0} with id {1}'.format(JOBNAME, id_job))

    print('    submitted job executes:')
    print('\n'.join(['>>>> ' + e for e in execthis.split('\n')]))
    print()


def submit_on_ubuntu(data, model, layers, xai_method, label, index, explanationdir):
    """ Submit a job on an ubuntu workstation. """

    print("Starting job {}".format(index))
    compute_relevances_for_class(data['datapath'],
                                 data['dataname'],
                                 data['datasetobject'],
                                 data['partition'],
                                 data['batchsize'],
                                 model['modelpath'],
                                 model['modelname'],
                                 layers,
                                 xai_method,
                                 label,
                                 explanationdir)

    print("Finished job {}".format(index))


def evaluate(filepath):
    """ Read config.yaml file and start evaluation. """

    with open(filepath) as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)

        print(configs)
        backend = configs['system_config']['backend']

        datapath = configs['data']['datapath']
        dataname = configs['data']['dataname']
        dataset = configs['data']['datasetobject']
        partition = configs['data']['partition']
        batchsize = configs['data']['batchsize']

        modelpath = configs['model']['modelpath']
        modelname = configs['model']['modelname']
        modeltype = configs['model']['modeltype']

        logdir = configs['dirs']['logdir']
        scriptdir = configs['dirs']['scriptdir']
        explanationdir = configs['dirs']['explanationdir']
        outputdir = configs['dirs']['outputdir']

        layers = configs['layers']

        xai_methods = configs['xai_methods']

        classes = configs['classes']

        quantifications = configs['quantifications']

        # configure dirs
        for dir in configs['dirs'].values():
            if not os.path.isdir(dir):
                print('Creating {}'.format(dir))
                os.makedirs(dir)

        # configure main args
        base_args = ""
        base_args += "-d " + datapath
        base_args += " -dn " + dataname
        base_args += " -dl " + dataset
        base_args += " -p " + partition
        base_args += " -bs " + str(batchsize)

        base_args += " -rd " + explanationdir
        # base_args += " -o " + outputdir  # + "separability/"

        base_args += " -m " + modelpath
        base_args += " -mn " + modelname

        dataset = get_dataset(dataset)
        dataset = dataset(datapath, partition)
        print(len(dataset))

        if classes[0] == "all":
            classes = dataset.classes

        # add quantification specific arguments
        if quantifications[0] == "relevance_computation":

            # base_args += " -o " + join_path(outputdir, "separability")

            # dataset = get_dataset(dataset)
            # dataset = dataset(datapath, partition)
            #
            # print(len(dataset))
            #
            # if classes[0] == "all":
            #     classes = dataset.classes

            job_size = 1000  # number of images to process per job
            job_index = 0

            for xai_method in xai_methods:
                print(xai_method)
                for label in classes:
                    print(label)
                    label_idx = dataset.classname_to_idx(label)

                    args = base_args + " -cl " + str(label_idx)
                    args = args + " -r " + xai_method
                    args = args + " -l " + ":".join(layers)

                    if backend == "sge":
                        for i in range(math.ceil(len(dataset) / job_size)):
                            job_args = args + " -si " + str(i * job_size) + " -ei " + str((i + 1) * job_size)

                            submit_on_sungrid(job_args, configs, configs, "relevance_computation", job_index)       # ToDo
                            job_index += 1

                    elif backend == "ubuntu":
                        submit_on_ubuntu(configs['data'], configs['model'], configs['layers'], xai_method, label_idx, job_index, explanationdir)
                        job_index += 1

        else:
            for quantification_dict in quantifications:

                quantification = list(quantification_dict.keys())[0]
                print(quantification)

                method_output_dir = join_path(outputdir, quantification)
                method_args = base_args + " -o " + method_output_dir

                if not os.path.isdir(method_output_dir):
                    print('Creating {}'.format(method_output_dir))
                    os.makedirs(method_output_dir)

                job_index = 0
                for xai_method in xai_methods:
                    method_args = method_args + " -r " + xai_method

                    if quantification != "one_class_separability":

                        for name in classes:
                            idx = dataset.classname_to_idx(name)

                            job_args = method_args + " -l " + layers[0]
                            job_args = job_args + " -cl " + str(idx)

                            if quantification == "pixelflipping":

                                job_args = job_args + " -pd " + quantification_dict[quantification]["args"]["distribution"]

                            # elif quantification == "model_parameter_randomization":
                            #
                            #     job_args = job_args + " -l " + layers[0]
                            #
                            # elif quantification == "one_class_separability":
                            #
                            #     job_args = job_args + " -l " + ":".join(layers)

                            # submit
                            if backend == "sge":
                                submit_on_sungrid(job_args, configs, quantification_dict[quantification]["config"],
                                                  quantification, job_index)
                                job_index += 1

                    else:
                        job_args = method_args + " -l " + layers[0]

                        if backend == "sge":
                            submit_on_sungrid(job_args, configs, quantification_dict[quantification]["config"],
                                              quantification, job_index)
                            job_index += 1


if __name__ == "__main__":
    print("running evaluation with config file {}".format(sys.argv[1]))
    if sys.argv[1].endswith(".yaml"):
        evaluate(sys.argv[1])
