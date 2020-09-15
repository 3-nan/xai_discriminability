#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=franz.motzkus@hhi.fraunhofer.de
#SBATCH --job-name=separability
#SBATCH --output=%j_%x.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

fname=$(basename "$1" .yaml)

source "/etc/slurm/local_job_dir.sh"
mkdir -p ${LOCAL_JOB_DIR}/job_results

tar -czf ${HOME}/pipeline/additional_files.tgz ${HOME}/pipeline/additional_files
echo "TAR DONE"
cp -r ${HOME}/pipeline/additional_files.tgz ${LOCAL_JOB_DIR}
echo "COPIED"
dir ${LOCAL_JOB_DIR}
tar -C ${LOCAL_JOB_DIR} -zxf ${LOCAL_JOB_DIR}/additional_files.tgz home/fe/motzkus/pipeline/additional_files --strip-components=4
dir ${LOCAL_JOB_DIR}
echo "EXTRACTED"

singularity run --nv \
	--bind ${LOCAL_DATA}/datasets/cifar10:/mnt/dataset \
	--bind ${LOCAL_JOB_DIR}/additional_files:/mnt/additional_files \
	--bind ${LOCAL_JOB_DIR}/job_results:/mnt/output \
	separability.sif \
	/mnt/additional_files/parameters/${fname}.yaml

cd ${LOCAL_JOB_DIR}
tar -czf ${fname}-${SLURM_JOB_ID}.tgz job_results
cp ${fname}-${SLURM_JOB_ID}.tgz ${SLURM_SUBMIT_DIR}
rm -rf ${LOCAL_JOB_DIR}/job_results
