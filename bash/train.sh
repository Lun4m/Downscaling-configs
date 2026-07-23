#!/bin/bash
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=standard-g
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --exclusive

set -e
module purge
module load cray-python
module load LUMI/25.03 partition/G
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SINGULARITYENV_PREPEND_PATH=/user-software/bin

# Runtime tweaks
export HYDRA_FULL_ERROR=1
# export AMD_SERIALIZE_KERNEL=3

# export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.5
CPU_BIND="mask_cpu:7e000000000000,7e00000000000000"
CPU_BIND="${CPU_BIND},7e0000,7e000000"
CPU_BIND="${CPU_BIND},7e,7e00"
CPU_BIND="${CPU_BIND},7e00000000,7e0000000000"

config_name=$(basename "$1")
config_path=$(dirname "$(readlink -f "$1")")

OVERRIDES=(
    #"diagnostics.log.mlflow.enabled=false"
)
# Parse additional overrides from SLURM_OVERRIDES environment variable
if [ -n "${SLURM_OVERRIDES:-}" ]; then
    echo "Parsing SLURM_OVERRIDES: $SLURM_OVERRIDES"
    OVERRIDES+=($SLURM_OVERRIDES)
fi

srun --cpu-bind=$CPU_BIND suv run anemoi-training train \
    --config-name "$config_name" \
    --config-path "$config_path" \
    "${OVERRIDES[@]}"
