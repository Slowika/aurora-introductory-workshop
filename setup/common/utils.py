"""Workshop setup utility functions."""

import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Component, Data, Model
from azure.ai.ml.operations import ComponentOperations, DataOperations, ModelOperations
from azure.identity import DefaultAzureCredential


def get_aml_ci_env_vars() -> tuple[str, str, str]:
    """Get AML compute instance env vars needed for storage and AML interfaces.

    Returns
    -------
    sub_id : str
        Azure subscription ID
    rg_name : str
        Azure resource group name
    ws_name : str
        Azure ML workspace name

    Raises
    ------
    KeyError
        If any of the required environment variables are not set.

    """
    sub_id = os.environ["MLFLOW_TRACKING_URI"].split("/")[6]
    rg_name = os.environ["CI_RESOURCE_GROUP"]
    ws_name = os.environ["CI_WORKSPACE"]
    return sub_id, rg_name, ws_name


def get_local_env_vars() -> tuple[str, str, str]:
    """Get local env vars needed for storage and AML interfaces.

    Returns
    -------
    sub_id : str
        Azure subscription ID
    rg_name : str
        Azure resource group name
    ws_name : str
        Azure ML workspace name

    Raises
    ------
    KeyError
        If any of the required environment variables are not set.

    """
    sub_id = os.environ["SUBSCRIPTION_ID"]
    rg_name = os.environ["RESOURCE_GROUP_NAME"]
    ws_name = os.environ["WORKSPACE_NAME"]
    return sub_id, rg_name, ws_name


def create_mlclient(*, local: bool) -> MLClient:
    """Return an authenticated MLClient for the current compute instance environment.

    Returns
    -------
    azure.ai.ml.MLClient
        Authenticated MLClient.

    """
    if local:
        sub_id, rg_name, ws_name = get_local_env_vars()
    else:
        sub_id, rg_name, ws_name = get_aml_ci_env_vars()
    return MLClient(DefaultAzureCredential(), sub_id, rg_name, ws_name)


def get_latest_asset(
    operations: ComponentOperations | DataOperations | ModelOperations,
    name: str,
) -> Component | Data | Model:
    """Return the latest version of a given asset.

    Parameters
    ----------
    operations : azure.ai.ml.operations.ComponentOperations |
        azure.ai.ml.operations.DataOperations | azure.ai.ml.operations.ModelOperations
        Operations interface for the asset type.
    name : str
        Name of the asset.

    Returns
    -------
    azure.ai.ml.entities.Component | azure.ai.ml.entities.Data |
        azure.ai.ml.entities.Model
        Latest version of the asset.

    """
    return next(iter(operations.list(name=name)))
