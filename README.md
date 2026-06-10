# Aurora Introductory Workshop

This repository comprises resources for the [Aurora](https://github.com/microsoft/aurora) fine-tuning on [Azure Machine Learning (AML)](https://learn.microsoft.com/en-us/azure/machine-learning/overview-what-is-azure-machine-learning?view=azureml-api-2) workshop.

## Repository structure

```md
aurora_introductory_workshop/
├── notebooks/
│   ├── 0_aurora_workshop.ipynb: core workshop Jupyter notebook for running jobs **on AML**
│   ├── 0_aurora_workshop_local.ipynb: core workshop Jupyter notebook for running jobs **locally**
│   ├── finetune_configs.yaml: configuration for fine-tuning jobs
│   └── inference_configs.yaml: configuration for inference jobs
├── setup/:
│   ├── common/: workshop setup helper logic and constants
│   ├── components/
│   │   ├── common/: Aurora helper constants, configuration models, and logic
│   │   ├── inference/:
│   │   │   ├── component.yaml: AML inference and evaluation component definition
│   │   │   └── main.py: core Aurora inference and evaluation logic script with CLI interface for local and remote use
│   │   └── finetuning/:
│   │       ├── component.yaml: AML fine-tuning component definition
│   │       ├── main.py: core Aurora fine-tuning logic script with CLI interface for local and remote use
│   │       └── utils.py: fine-tuning helpers and utility logic
│   ├── environment/: definition for a single AML environment in which to run jobs
│   └── notebooks/: initial AML workspace setup Jupyter notebooks
│       ├── load_era5.ipynb: load a subset of ERA5 data into Azure Blob Storage and register it as a data asset within an AML workspace
│       ├── load_era5_local.ipynb: load a subset of ERA5 data into local storage
│       └── load_model.ipynb: download a specified Aurora variant checkpoint from Hugging Face and optionally register it as a model asset within an AML workspace
├── tests/: unit and integration tests
└── .template.env: contains environment variables to be set for local interaction with an AML workspace
```

## Environment variables

Three environment variables are required for local use (SDK-based AML entity deployment, AML workspace setup notebook execution, and AML job submission):

- `SUBSCRIPTION_ID`: ID of the subscription in which resources exist
- `RESOURCE_GROUP_NAME`: name of the resource group in which resources exist
- `WORKSPACE_NAME`: name of the Azure Machine Learning workspace to use

These can be set permanently via .bashrc or the Windows environment variable manager or temporarily through a .env file. See [.template.env](.template.env) for instructions. Enable VSCode's `python.terminal.useEnvFile` setting to inject values into terminals or your preferred method for exposing project values as environment variables (e.g. `dotenv`).

Three environment variables are required for remote use but are automatically set in Azure Machine Learning compute instances:

- `AZUREML_CR_AZUREML_CONTEXT`: used to obtain the ID of the subscription, name of the resource group, and name of the AML workspace in which resources exist and jobs should be run

## Setup and Usage

Provided hardware is sufficient, workshop code can be run in local and remote (AML) environments. Key project settings are described in [setup/common/constants.py](setup/common/constants.py) - **do not edit** those marked as such. For settings not marked as non-editable, it's recommended to amend these in-place to propagate changes across workshop resources. Other settings found elsewhere include:

| Location | Setting | Description | Editable |
| --- | --- | --- | --- |
| [setup/notebooks/load_era5_local.ipynb](setup/notebooks/load_era5_local.ipynb) / [setup/notebooks/load_era5](setup/notebooks/load_era5.ipynb) | `EXTRA_SFC_VARS`, `EXTRA_ATMOS_VARS`, `EXTRA_LEVELS` | Additional non-standard variables and levels to load | Yes |
| [setup/components/common/constants.py](setup/components/common/constants.py) | `SURF_VAR_MAP`, `STATIC_VAR_MAP`, `ATMOS_VAR_MAP` | ERA5 longname -> Aurora shortname map for surface, static, and atmospheric variables | No, specific to expected input data and checkpoint |
| | `ATMOS_LEVELS` | Aurora-expected pressure levels | No, specific to expected input data and checkpoint |

Other Aurora variants and their checkpoints are available on [Hugging Face](https://huggingface.co/microsoft/aurora/tree/main) though many, if not all, will not work out-of-the-box with this workshop. Different models have different input data expectations and pre-configured data loading is specific to `aurora-0.25-pretrained.ckpt`. Model loading ([`setup.components.common.utils.load_model`](setup/components/common/utils.py#45)) is also specific to `AuroraPretrained`.

A hybrid execution model is possible in that all workspace setup (entity deployment and asset creation) and job submission can be run locally with the jobs themselves executed in AML.

### Run Configurations

AML job or `runpy` script executions rely on [inference](notebooks/inference_configs.yaml) and [finetuning](notebooks/finetune_configs.yaml) YAML configurations. Add new or update run configurations using the provided examples and [`pydantic` model definitions](setup/components/common/models.py) as guides.

### Local

These instructions describe completely local execution of inference and fine-tuning. To do so, capable hardware is required. Inference can run on CPU, albeit slowly. Fine-tuning of the full Aurora 0.25 degree pre-trained model as in this workshop requires an A100, H100, or equivalent GPU. Also required is sufficient storage to load the model and data. The [`uv`](https://docs.astral.sh/uv/) package manager is the easiest way to get started.

1. Create a Python virtual environment at the repository root and install dependencies (optionally including the `dev` group for tests)
2. Run the [data](setup/notebooks/load_era5_local.ipynb) and [model](setup/notebooks/load_model.ipynb) loading Jupyter notebooks in [setup/notebooks](setup/notebooks/) using the virtual environment as the kernel, skipping the final cell of the latter notebook to avoid remote registration of the model asset
3. Run inference and fine-tuning with [notebooks/0_aurora_workshop_local.ipynb](notebooks/0_aurora_workshop_local.ipynb)

### Remote (AML)

These instructions describe completely remote (AML) execution of inference and fine-tuning with deployment of some assets from a local environment. As [above](#locally), GPU-enabled compute is required for fine-tuning.

For SDK-based entity deployment or local execution of the data loading, model loading, or workshop notebooks, set [environment variables](#environment-variables).

1. Load the data and model loading Jupyter notebooks (and their local first-party dependencies) to the Notebooks tab of the AML Studio UI and run them with the pre-built `Python 3.10 - SDK v2` kernel:
    - The easiest way to make notebooks and 1P dependencies available on AML is via `git clone` on a [compute instance terminal](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-access-terminal?view=azureml-api-2)
    - Notebooks are best run on a CPU compute instance with ~64 GB of memory (at least for data loading)
    - The data and model are registered as workspace assets in their respective final notebook cells
2. Deploy AML component and environment entities to the workspace:
    - Components (for each in [setup/components](setup/components)):
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
    - Environments ([setup/environment](setup/environment/)):
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
3. Run inference and fine-tuning with [notebooks/0_aurora_workshop.ipynb](notebooks/0_aurora_workshop.ipynb):
    - Can be run locally (with [environment variables](#environment-variables) set) or via the AML Studio UI Notebooks tab, the latter with the pre-built `Python 3.10 - SDK v2` kernel
    - Target compute is selected in the fourth code cell with `CLUSTER_NAME = next(iter(ml_client.compute.list(compute_type="amlcompute"))).name`. This assumes the target is a compute cluster, of which there is just one in the workspace. While this was true for the workshop instance, should this not be the case or you wish to select compute (instance or cluster) by name, replace with `CLUSTER_NAME = ml_client.compute.get({compute_name}).name`, where {compute_name} is the name of the compute to run inference and / or fine-tuning on.
    - **Do not** alter any notebook values other than the aforementioned (if necessary), use input boxes that appear where prompted to specify a participant ID and job configurations - carefully read each cell's markdown description

## Testing

Tests require `pytest`.

Integration tests in which inference and fine-tuning are tested end-to-end require the `aurora.AuroraSmallPretrained` checkpoint. Download it to the tests/models/ (gitignored) directory with:
```bash
huggingface-cli download microsoft/aurora aurora-0.25-small-pretrained.ckpt --local-dir tests/models/
```
Or manually from: https://huggingface.co/microsoft/aurora/blob/main/aurora-0.25-small-pretrained.ckpt
