### Get the stuff
Clone this repo in your home directory on LUMI
```term
git clone git@github.com:Lun4m/Downscaling-configs.git
cd Downscaling-configs
```

### Setup environment variables 

Add the following to your `~/.bashrc`, substituting xxxxxxxxx with your project
number. Projects on LUMI are named `project_xxxxxxxxx`, where xxxxxxxxx is the
number needed in this step.

```bash
# Project settings
export PROJ_NUM=xxxxxxxxx # your project number
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
repodir="$HOME/Downscaling-configs"
if [[ -d "$repodir/bin" ]] && [[ ":$PATH:" != *":$repodir/bin:"* ]]; then
    export PATH="$repodir/bin:$PATH"
fi
```

Then run
```term
source ~/.bashrc
```

### Install uv
```term
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Optionally install anemoi
If you need to do development on Anemoi, install anemoi
packages locally
```term
bash script/download_anemoi.sh ~/anemoi-packages
```
The path to the packages needs to be exported in the `[tool.uv.sources]`
section of `pyproject.toml`.

### Build virtual environment
```term
build-lumi-venv $CONTAINERS_DIR/container.sif
```
The container name changes depending on Python and Pytorch versions,
so first check which containers are available in `$CONTAINERS_DIR`.

> [!CAUTION]
> The environments live in a shared directory in `$SUV_VENV_LOCATION`.
>
> Under very specific circumstances, it might happen to be prompted to
> rebuild an environment even though the active user never built one in the
> first place. In this situation, the solution is to simply first run
> ```term 
> rm .venv-name
> ```
> and proceed with the `build-lumi-venv` command.

### MLFlow setup
Authenticate to the MLFlow logging server
```term
suv anemoi-training mlflow login --url https://mlflow.ecmwf.int
```
It's also important to change the `project_name` and `experiment_name` under
`log.mlflow` in the `diagnostics/downscaling.yaml` config file.

> [!NOTE]
> If you can't login to the server because you don't have an ECMWF account,
> you can disable online logging in your yaml config:
> ```yaml
> diagnostics:
>   log:
>     mlflow:
>       offline: true
> ```


### Traning
```term
sbatch --account=$PROJ \
        --job-name=my-beautiful-training \
        --nodes=8 \
        --time=02:00:00 \
        scripts/train.sh \
        carra_east.yaml
```
