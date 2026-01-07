"""Aurora inference Azure ML Component definition."""

from azure.ai.ml import Input, Output
from azure.ai.ml.entities import CommandComponent

from setup.common.utils import create_mlclient

if __name__ == "__main__":
    component = CommandComponent(
        name="workshop_aurora_inference",
        description="Component for performing inference with Aurora.",
        version="1",
        command=(
            "python -m main "
            "--model ${{inputs.model}} "
            "--data ${{inputs.data}} "
            "--steps ${{inputs.steps}} "
            "--predictions ${{outputs.predictions}}"
        ),
        code="./setup/components/inference",
        # NOTE: environment does get persisted in registered component
        environment="aurora-inference:1",
        # NOTE: mode and path are stripped from inputs and outputs on registration
        inputs={
            "model": Input(
                type="custom_model",
                description="Pretrained Aurora model checkpoint.",
            ),
            "data": Input(
                type="uri_folder",
                optional=True,
                description=(
                    "Data asset containing the initial state for inference. Leave "
                    "empty for a dry run with dummy data."
                ),
            ),
            "steps": Input(
                type="integer",
                default=1,
                min=1,
                max=10,
                description="Number of autoregressive steps to perform.",
            ),
        },
        outputs={
            "predictions": Output(
                type="uri_folder",
                description="Output predictions.",
            ),
        },
        instance_count=1,
        is_deterministic=True,
        additional_includes=["./setup/components/common/"],
    )
    mlc = create_mlclient(local=True)
    mlc.components.create_or_update(component)
