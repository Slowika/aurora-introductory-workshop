"""Workshop setup utility functions."""

import json
import os
from typing import overload

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Component, Data, Model
from azure.ai.ml.operations import ComponentOperations, DataOperations, ModelOperations
from azure.core.exceptions import ResourceNotFoundError
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

    """
    azureml_ctx = os.environ["AZUREML_CR_AZUREML_CONTEXT"]
    azureml_ctx_dict = json.loads(azureml_ctx)
    sub_id = azureml_ctx_dict["subscription_id"]
    rg_name = azureml_ctx_dict["resource_group"]
    ws_name = azureml_ctx_dict["workspace_name"]
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


@overload
def get_latest_asset(
    operations: ComponentOperations,
    name: str,
) -> Component: ...


@overload
def get_latest_asset(
    operations: DataOperations,
    name: str,
) -> Data: ...


@overload
def get_latest_asset(
    operations: ModelOperations,
    name: str,
) -> Model: ...


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

    Raises
    ------
    azure.core.exceptions.ResourceNotFoundError
        If no asset is found with the given name.

    """
    try:
        return next(iter(operations.list(name=name)))
    except StopIteration as e:
        msg = f"Asset not found: name={name}, type={operations.__class__.__name__}"
        raise ResourceNotFoundError(msg) from e
