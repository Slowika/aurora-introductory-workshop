# Aurora Introductory Workshop

This repository comprises resources for the Aurora fine-tuning on Azure Machine Learning (AML) workshop.

## Repository structure

```md
aurora_introductory_workshop/
├── notebooks/
│   └── 0_aurora_workshop.ipynb: core workshop Jupyter notebook for running jobs on AML.
├── setup/: core Aurora logic and AML workspace setup resources
│   ├── common/: AML workspace setup helper logic
│   ├── components/
│   │   ├── common/: Aurora helper constants and logic
│   │   ├── inference/:
|   │   │   ├── component.py: AML inference component definition and deployment
|   │   │   └── main.py: core Aurora inference logic script with CLI interface for local and remote environments
|   |   └── training/:
|   │       ├── component.py: AML fine-tuning component definition and deployment
|   │       └── main.py: core Aurora fine-tuning logic script with CLI interface for local and remote environments
│   ├── environments/: definitions for AML environments in which to run Aurora.
│   └── notebooks/: initial workspace setup Jupyter notebooks.
│       ├── load_era5.ipynb: load a subset of ERA5 data into Azure Blob Storage and register it as a data asset within the workspace.
│       └── register_model.ipynb: download a specified Aurora variant checkpoint from Hugging Face and register it as a model asset within the workspace.
└── .template.env: contains environment variables to be set for local use, particularly relevant to workspace setup. Copy to a new file named `.env` and fill values.
```

## Environment variables

Three environment variables are required for local use:

- SUBSCRIPTION_ID: ID of the subscription in which resources exist
- RESOURCE_GROUP_NAME: name of the resource group in which resources exist
- WORKSPACE_NAME: name of the Azure Machine Learning workspace to use

These can be set permanently via .bashrc or the Windows environment variable manager or temporarily through a `.env` file. See `./.template.env`.

Three environment variables are required for remote use but are automatically set in Azure Machine Learning compute instances:
- MLFLOW_TRACKING_URI: used to extract the ID of the subscription in which resources exist
- CI_RESOURCE_GROUP: name of the resource group in which resources exist
- CI_WORKSPACE: name of the Azure Machine Learning workspace to use
