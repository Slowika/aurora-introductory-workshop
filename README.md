# Aurora Introductory Workshop

This repository comprises resources for the [Aurora](https://github.com/microsoft/aurora) fine-tuning on Azure Machine Learning (AML) workshop.

## Repository structure

```md
aurora_introductory_workshop/
├── notebooks/
│   ├── 0_aurora_workshop.ipynb: core workshop Jupyter notebook for running jobs on AML
│   ├── finetune_configs.yaml: configuration for fine-tuning jobs
│   └── inference_configs.yaml: configuration for inference jobs
├── setup/:
│   ├── common/: AML workspace setup and interaction helper logic
│   ├── components/
│   │   ├── common/: Aurora helper constants, configuration models, and logic
│   │   ├── inference/:
│   │   │   ├── component.yaml: AML inference and evaluation component definition
│   │   │   └── main.py: core Aurora inference and evaluation logic script with CLI interface for local and remote use
│   │   └── training/:
│   │       ├── component.yaml: AML fine-tuning component definition
│   │       ├── main.py: core Aurora fine-tuning logic script with CLI interface for local and remote use
│   │       └── utils.py: fine-tuning helpers and utility logic
│   ├── environment/: definition for a single AML environment in which to run Aurora.
│   └── notebooks/: initial AML workspace setup Jupyter notebooks
│       ├── load_era5.ipynb: load a subset of ERA5 data into Azure Blob Storage and register it as a data asset within an AML workspace
│       └── register_model.ipynb: download a specified Aurora variant checkpoint from Hugging Face and register it as a model asset within an AML workspace
├── tests/: unit and integration tests
└── .template.env: contains environment variables to be set for local use of some resources
```

## Environment variables

Three environment variables are required for local use (SDK entity deployment and setup notebook execution):

- `SUBSCRIPTION_ID`: ID of the subscription in which resources exist
- `RESOURCE_GROUP_NAME`: name of the resource group in which resources exist
- `WORKSPACE_NAME`: name of the Azure Machine Learning workspace to use

These can be set permanently via .bashrc or the Windows environment variable manager or temporarily through a `.env` file. See `./.template.env` for instructions. Enable VSCode's `python.terminal.useEnvFile` setting to inject `.env` values into terminals or your preferred method for exposing project values as environment variables

Three environment variables are required for remote use but are automatically set in Azure Machine Learning compute instances:
- `MLFLOW_TRACKING_URI`: used to extract the ID of the subscription in which resources exist
- `CI_RESOURCE_GROUP`: name of the resource group in which resources exist
- `CI_WORKSPACE`: name of the Azure Machine Learning workspace to use

## Setup

- Use `setup/notebooks` to load and register the data and models:
    - Best done in AML notebooks with a CPU compute instance with ~64 GB of memory (at least for data loading) to avoid downloading then uploading data
- Deploy AML component and environment entities:
    - Components (`setup/components`):
        1. Studio: from the Components tab, select "New Component" and follow the component creation dialogue
        2. CLI: `az ml component create -f setup/components/<name>/component.yaml -w {workspace-name} -g {resource-group-name}`
        3. SDK:
        ```python
        from azure.ai.ml import MLClient, load_component
        from azure.identity import DefaultAzureCredential

        component = load_component("setup/components/{name}/component.yaml")
        client = MLClient(
           credential=DefaultAzureCredential(),
           subscription_id={subscription_id},
           resource_group_name={resource_group_name},
           workspace_name={workspace_name},
        )
        client.components.create_or_update(component)
        ```
    - Environments (`setup/environments`):
        1. Studio: from the Environments tab, select "Create" and follow the environment creation dialogue
        2. CLI: `az ml environment create -f setup/environment/environment.yaml -w {workspace-name} -g {resource-group-name}`
        3. SDK:
        ```python
        from azure.ai.ml import MLClient, load_environment

        component = load_environment("setup/environments/environment.yaml")
        client = MLClient(
           credential=DefaultAzureCredential(),
           subscription_id={subscription_id},
           resource_group_name={resource_group_name},
           workspace_name={workspace_name},
        )
        client.environments.create_or_update(component)
        ```
- For SDK-based entity deployment or local setup notebook execution:
    - See [environment variables](#environment-variables)
    - Create a Python virtual environment or conda environment with the dependencies described in `pyproject.toml` to run setup code
- In `notebooks/0_aurora_workshop.ipynb`, the compute cluster is selected in the fourth code cell with `CLUSTER_NAME = next(iter(ml_client.compute.list(compute_type="amlcompute"))).name`. This assumes there is one compute cluster in the workspace, which was true for the workshop instance. Should this not be the case, or you want to select compute (instance or cluster) by name, replace with `CLUSTER_NAME = ml_client.compute.get("name").name`, where "name" is the name of the compute you wish to use

## Testing

Tests require `pytest`.

Integration tests in which inference and fine-tuning are tested end-to-end require the `aurora.AuroraSmallPretrained` checkpoint. Download it to the `tests/models/` directory with:
```bash
huggingface-cli download microsoft/aurora aurora-0.25-small-pretrained.ckpt --local-dir tests/models/
```
Or manually from: https://huggingface.co/microsoft/aurora/blob/main/aurora-0.25-small-pretrained.ckpt
