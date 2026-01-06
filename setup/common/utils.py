"""Workshop setup utility functions."""

import os

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


def get_aml_ci_env_vars() -> tuple[str, str, str]:
    """Get AML compute instance env vars needed for storage and AML interfaces.

    Returns
    -------
    tuple[str, str, str]
        Subscription ID, resource group name, AML workspace name.

    Raises
    ------
    KeyError
        If any of the required environment variables are not set.

    """
    sub_id = os.environ["MLFLOW_TRACKING_URI"].split("/")[6]
    rg_name = os.environ["CI_RESOURCE_GROUP"]
    ws_name = os.environ["CI_WORKSPACE"]
    return sub_id, rg_name, ws_name


def create_mlclient() -> MLClient:
    """Return an authenticated MLClient for the current compute instance environment.

    Returns
    -------
    azure.ai.ml.MLClient
        Authenticated MLClient.

    """
    return MLClient(DefaultAzureCredential(), *get_aml_ci_env_vars())
