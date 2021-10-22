import os
import sys
import math
import time
import yaml
import subprocess

from xaitestframework.experiments.attribution_computation import compute_attribution_wrapper
from xaitestframework.dataloading.custom import get_dataset


def submit_on_sungrid(args, configs, jobconfig, quantification_method, index, with_attr=None):
    """ Submit the specified job on a sungrid engine. """

    print('Generating and submitting jobs for:')
    print('    {}'.format(args))

    # generate script
    execthis = ['#!/bin/bash']
    execthis += ['hostname']
    execthis += ['mkdir -p /data/local/users/motzkus']
    execthis += ['mkdir /data/local/users/motzkus/$JOB_ID']

    execthis += ['cp /data/cluster/users/motzkus/data/{}.tar.xz /data/local/users/motzkus/$JOB_ID/'.format(configs["data"]["dataname"])]
    execthis += ['tar -xf /data/local/users/motzkus/$JOB_ID/{}.tar.xz -C /data/local/users/motzkus/$JOB_ID/'.format(configs["data"]["dataname"])]

    if with_attr:
        xai_method, layer, classidx = with_attr
        if classidx:
            execthis += ['tar -zcf /data/cluster/users/motzkus/data_$JOB_ID.tar.gz -C /data/cluster/users/motzkus/results/ {}_{}/{}/{}/{}'.format(configs["data"]["dataname"], configs["model"]["modelname"], xai_method, layer, classidx)]
        else:
            execthis += ['tar -zcf /data/cluster/users/motzkus/data_$JOB_ID.tar.gz -C /data/cluster/users/motzkus/results/ {}_{}/{}/{}'.format(configs["data"]["dataname"], configs["model"]["modelname"], xai_method, layer)]

        execthis += ['cp /data/cluster/users/motzkus/data_$JOB_ID.tar.gz /data/local/users/motzkus/$JOB_ID/']
        execthis += ['tar -xf /data/local/users/motzkus/$JOB_ID/data_$JOB_ID.tar.gz -C /data/local/users/motzkus/$JOB_ID/']
        execthis += ['rm /data/cluster/users/motzkus/data_$JOB_ID.tar.gz']

    execthis += ['mkdir /data/local/users/motzkus/$JOB_ID/results']
    execthis += ['source /home/fe/motzkus/.bashrc']  # enables conda for bash ToDo: change dir
    execthis += ['{} activate {}'.format(configs['system_config']['conda'], configs['system_config']['environment'])]
    execthis += ['python3 -m xaitestframework.experiments.{} {} -x /data/local/users/motzkus/$JOB_ID'.format(quantification_method, args)]
    execthis += ['{} deactivate'.format(configs['system_config']['conda'])]  # leave venv
    execthis += ['tar -zcf /data/local/users/motzkus/$JOB_ID/results_$JOB_ID.tar.gz -C /data/local/users/motzkus/$JOB_ID/results .']

    if with_attr:
        execthis += ['cp /data/local/users/motzkus/$JOB_ID/results_$JOB_ID.tar.gz /data/cluster/users/motzkus/results/']
        execthis += [
            'tar -xf /data/cluster/users/motzkus/results/results_$JOB_ID.tar.gz -C /data/cluster/users/motzkus/results/' + quantification_method]

    else:
        execthis += ['cp /data/local/users/motzkus/$JOB_ID/results_$JOB_ID.tar.gz /data/cluster/users/motzkus/results/']
        execthis += ['tar -xf /data/cluster/users/motzkus/results/results_$JOB_ID.tar.gz -C /data/cluster/users/motzkus/results/']

    execthis += ['rm /data/cluster/users/motzkus/results/results_$JOB_ID.tar.gz']
    execthis += ['rm -r /data/local/users/motzkus/$JOB_ID']
    execthis = '\n'.join(execthis)

    JOBNAME = '[{}]_of_"{}"'.format(index + 1, quantification_method)
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
    compute_attribution_wrapper(data['datapath'],
                                data['dataname'],
                                data['datasetobject'],
                                data['partition'],
                                data['batchsize'],
                                model['modelpath'],
                                model['modelname'],
                                model['modeltype'],
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
        base_args += " -mt " + modeltype

        dataset_class = get_dataset(dataset)
        dataset = dataset_class(datapath, partition)
        print(len(dataset))

        if classes[0] == "all":
            classes = dataset.classes

        # add quantification specific arguments
        if list(quantifications[0].keys())[0] == "attribution_computation":

            job_size = 500   # 1000  # number of images to process per job
            job_index = 0

            for label in classes:
                print(label)
                label_idx = dataset.classname_to_idx(label)

                class_dataset = dataset_class(datapath, partition, classidx=[label_idx])

                for xai_method in xai_methods:
                    print(xai_method)

                    args = base_args + " -cl " + str(label_idx)
                    args = args + " -r " + xai_method
                    args = args + " -l " + ":".join(layers)

                    if backend == "sge":
                        for i in range(math.ceil(len(class_dataset) / job_size)):
                            job_args = args + " -si " + str(i * job_size) + " -ei " + str((i + 1) * job_size)

                            submit_on_sungrid(job_args,
                                              configs,
                                              quantifications[0]["attribution_computation"]["config"],
                                              "attribution_computation",
                                              job_index)
                            job_index += 1

                    elif backend == "ubuntu":
                        submit_on_ubuntu(configs['data'],
                                         configs['model'],
                                         configs['layers'],
                                         xai_method,
                                         label_idx,
                                         job_index,
                                         explanationdir)
                        job_index += 1

        else:
            for quantification_dict in quantifications:

                quantification = list(quantification_dict.keys())[0]
                print(quantification)

                method_output_dir = os.path.join(outputdir, quantification)
                method_args = base_args + " -o " + method_output_dir

                if not os.path.isdir(method_output_dir):
                    print('Creating {}'.format(method_output_dir))
                    os.makedirs(method_output_dir)

                job_index = 0
                for xai_method in xai_methods:
                    xai_args = method_args + " -r " + xai_method

                    if quantification in ["model_parameter_randomization", "pixelflipping", "separability", "manifold_outlier_pixelflipping_experiment", "activation_eval"]:

                        for name in classes:
                            idx = dataset.classname_to_idx(name)

                            # job_args = xai_args + " -l " + layers[0]
                            job_args = xai_args + " -cl " + str(idx)

                            if quantification == "pixelflipping" or quantification == "manifold_outlier_pixelflipping_experiment":

                                job_args = job_args + " -l " + layers[0]
                                job_args = job_args + " -pd " + quantification_dict[quantification]["args"]["distribution"]
                                percentages = ":".join([str(p) for p in quantification_dict[quantification]["args"]["percentages"]])
                                job_args = job_args + " -pv " + percentages

                                # submit
                                if backend == "sge":
                                    submit_on_sungrid(job_args, configs, quantification_dict[quantification]["config"],
                                                      quantification, job_index,
                                                      with_attr=(xai_method, layers[0], idx))
                                    job_index += 1

                            elif quantification == "separability":

                                for layer in layers:
                                    job_args_layer = job_args + " -l {}".format(layer)

                                    submit_on_sungrid(job_args_layer, configs,
                                                      quantification_dict[quantification]["config"],
                                                      quantification, job_index,
                                                      with_attr=(xai_method, layer, str(idx)))
                                    job_index += 1
                                    time.sleep(3)

                            elif quantification == "activation_eval":

                                job_args = job_args + " -pd " + quantification_dict[quantification]["args"]["distribution"]
                                percentages = ":".join([str(p) for p in quantification_dict[quantification]["args"]["percentages"]])
                                job_args = job_args + " -pv " + percentages

                                for layer in layers:
                                    job_args_2 = job_args + " -l " + layer

                                    submit_on_sungrid(job_args_2, configs, quantification_dict[quantification]["config"],
                                                      quantification, job_index,
                                                      with_attr=(xai_method, layer, str(idx)))
                                    job_index += 1

                            elif quantification == "model_parameter_randomization":

                                job_args = job_args + " -l " + layers[0]
                                if quantification_dict[quantification]["args"]["max_index"]:
                                    job_args = job_args + " -mi " + str(quantification_dict[quantification]["args"]["max_index"])
                                if quantification_dict[quantification]["args"]["distance_measures"]:
                                    job_args = job_args + " -dm " + ":".join(quantification_dict[quantification]["args"]["distance_measures"])
                                job_args = job_args + " -s {}".format(quantification_dict[quantification]["args"]["setting"])

                                # submit
                                if backend == "sge":
                                    submit_on_sungrid(job_args, configs, quantification_dict[quantification]["config"],
                                                      quantification, job_index,
                                                      with_attr=(xai_method, layers[0], idx))
                                    job_index += 1

                    else:
                        for layer in layers:
                            job_args = xai_args + " -l " + layer

                            if quantification == "pointing_game":
                                job_args = job_args + " -gb " + str(quantification_dict[quantification]["args"]["gaussian_blur"])

                            if quantification == "tsne_manifold" or quantification == "tsne_manifold_one_class":
                                job_args = job_args + " -pd " + quantification_dict[quantification]["args"][
                                    "distribution"]
                                percentages = ":".join(
                                    [str(p) for p in quantification_dict[quantification]["args"]["percentages"]])
                                job_args = job_args + " -pv " + percentages

                            if backend == "sge":
                                submit_on_sungrid(job_args, configs, quantification_dict[quantification]["config"],
                                                  quantification, job_index,
                                                  with_attr=(xai_method, layer, None))
                                job_index += 1


if __name__ == "__main__":
    print("running evaluation with config file {}".format(sys.argv[1]))
    if sys.argv[1].endswith(".yaml"):
        evaluate(sys.argv[1])
