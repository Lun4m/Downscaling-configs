#!/bin/bash
#SBATCH --job-name=infer210
#SBATCH --output=./logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

module purge
module load cray-python
module load LUMI/25.03 partition/G
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

config_name=$1
export SINGULARITYENV_PREPEND_PATH=/user-software/bin

# Runtime tweaks
export HYDRA_FULL_ERROR=1
export AMD_SERIALIZE_KERNEL=3

srun suv run anemoi-inference run \
    --defaults inference/post_processors.yaml \
    "$config_name"
