# Aurora Introductory Workshop

This repository comprises resources for the Aurora fine-tuning on Azure Machine Learning (AML) workshop.

## Repository structure

```md
aurora_introductory_workshop/
├── notebooks/
│   ├── 0_aurora_workshop.ipynb: core workshop Jupyter notebook for running jobs on AML
│   ├── finetune_configs.yaml: configuration for fine-tuning jobs
│   └── inference_configs.yaml: configuration for inference jobs
├── setup/: core Aurora logic and AML workspace setup resources
│   ├── common/: AML workspace setup helper logic
│   ├── components/
│   │   ├── common/: Aurora helper constants and logic
│   │   ├── inference/:
│   │   │   ├── component.py: AML inference and evaluation component definition and deployment
│   │   │   └── main.py: core Aurora inference and evaluation logic script with CLI interface for local and remote environments
│   │   └── training/:
│   │       ├── component.py: AML fine-tuning component definition and deployment
│   │       └── main.py: core Aurora fine-tuning logic script with CLI interface for local and remote environments
│   ├── environments/: definitions for AML environments in which to run Aurora.
│   └── notebooks/: initial workspace setup Jupyter notebooks
│       ├── load_era5.ipynb: load a subset of ERA5 data into Azure Blob Storage and register it as a data asset within the workspace
│       └── register_model.ipynb: download a specified Aurora variant checkpoint from Hugging Face and register it as a model asset within the workspace
└── .template.env: contains environment variables to be set for local use, particularly relevant to workspace setup. Copy to a new file named `.env` and fill values.
```

## Environment variables

Three environment variables are required for local use:

- `SUBSCRIPTION_ID`: ID of the subscription in which resources exist
- `RESOURCE_GROUP_NAME`: name of the resource group in which resources exist
- `WORKSPACE_NAME`: name of the Azure Machine Learning workspace to use

These can be set permanently via .bashrc or the Windows environment variable manager or temporarily through a `.env` file. See `./.template.env`.

Three environment variables are required for remote use but are automatically set in Azure Machine Learning compute instances:
- `MLFLOW_TRACKING_URI`: used to extract the ID of the subscription in which resources exist
- `CI_RESOURCE_GROUP`: name of the resource group in which resources exist
- `CI_WORKSPACE`: name of the Azure Machine Learning workspace to use

## Deploying resources for use

- Use `setup/notebooks` to load and register the data and models. This is best done in AML notebooks with a CPU compute instance with ~64 GB of memory, at least for data loading
- Use `setup/components/<component name>/component.py` to build and deploy each component to the workspace - this is currently configured to be done locally but you can change `mlc = create_mlclient(local=True)` to `local=False` (line 68 of `inference/component.py`, line 88 for `training`) to deploy from an AML compute instance
- Build an environment in AML with the Dockerfile in `setup/environments/aurora_minimal` by copying / uploading the Dockerfile via the Environments tab then Create button to open the environment creation dialogue
- If deploying anything from a local machine, follow the instructions in `./.template.env` file to create the necessary environment variables for local interaction with the AML workspace. You'll also need a Python virtual environment or conda environment with the dependencies described in `pyproject.toml` / `uv.lock` to run setup code
- In `notebooks/0_aurora_workshop.ipynb`, the compute cluster is selected in the fourth code cell with `CLUSTER_NAME = next(iter(ml_client.compute.list(compute_type="amlcompute"))).name`. This assumes there is one compute cluster in the workspace, which was true for the workshop instance. Should this not be the case, or you want to select compute (instance or cluster) by name, replace with `CLUSTER_NAME = ml_client.compute.get("name").name`, where "name" is the name of the compute you wish to use
