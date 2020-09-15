#!/bin/bash

for file in additional_files/parameters/*.yaml; do
  echo "Starting experiment ${file}"
  sbatch -p gpu batch_separability.sh ${file}
done
