#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=franz.motzkus@hhi.fraunhofer.de
#SBATCH --job-name=train_model
#SBATCH --output=%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

source "/etc/slurm/local_job_dir.sh"
mkdir -p ${LOCAL_JOB_DIR}/job_results

dir ${LOCAL_JOB_DIR}

singularity run --nv \
	--bind ${LOCAL_DATA}/datasets/cifar10:/mnt/dataset \
	--bind ${LOCAL_JOB_DIR}/job_results:/mnt/output \
	train_model.sif 

cd ${LOCAL_JOB_DIR}
tar -czf ${SLURM_JOB_ID}.tgz job_results
cp ${SLURM_JOB_ID}.tgz ${SLURM_SUBMIT_DIR}
rm -rf ${LOCAL_JOB_DIR}/job_results
