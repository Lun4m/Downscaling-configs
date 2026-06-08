1. Clone this repo in your home directory

1. Add the following to your `~/.bashrc`
    ```bash
    # Project settings
    export PROJ_NUM= # your project number
    export PROJ=project_${PROJ_NUM}
    export SCRATCH=/scratch/${PROJ}
    export PROJECT=/project/${PROJ}
    export FLASH=/flash/${PROJ}
    export CONTAINERS_DIR=$SCRATCH/containers
    export SUV_VENV_LOCATION=${PROJECT}/environment/venvs

    # LUMI-O settings
    export AWS_ENDPOINT_URL=https://lumidata.eu
    export AWS_DEFAULT_REGION=lumi-prod
    export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
    export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required

    # Export project binaries
    repodir="$HOME/repo-name" # repo-name is the name of your cloned repository
    if [[ -d "$repodir/lumi/bin" ]] && [[ ":$PATH:" != *":$repodir/lumi/bin:"* ]]; then
        export PATH="$repodir/lumi/bin:$PATH"
    fi
    ```
    Then
    ```term
    > source ~/.bashrc
    ```

1. Install uv
    ```term
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

1. Install anemoi packages
    ```term
    bash repo-name/lumi/download_anemoi.sh target_directory
    ```

1. Build virtual environment
    ```term
    cd repo-name/recipes
    build-lumi-venv sif_container
    ```

1. Run training
    ```term
    cd repo-name/recipes
    sbatch --account=$PROJ \
           --job-name=my-beautiful-training \
           --nodes=8 \
           --time=02:00:00 \
           ../lumi/scripts/train.sh \
           carra_east.yaml
    ```
