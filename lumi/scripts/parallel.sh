#!/bin/bash
#SBATCH --job-name=infer210
#SBATCH --output=./logs/%x_%j.out
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --hint=nomultithread

# The arguments to the script:
# 1) the template config
# 2) the runid of the checkpoint
# 3+) list of overrides to pass to anemoi-inference
config=$1
export RUNID=$2
shift 2
overrides=("$@")

module purge
module load cray-python
module load LUMI/25.03 partition/G
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SINGULARITYENV_PREPEND_PATH=/user-software/bin

# Runtime tweaks
export HYDRA_FULL_ERROR=1
export AMD_SERIALIZE_KERNEL=3

for override in "${overrides[@]}"; do
    echo "Submitting job with these overrides: $override"
    # Note that $override needs to be unquoted here
    srun --exclusive --nodes=1 --ntasks=1 \
        suv run anemoi-inference run "$config" "$override" &
done

wait
