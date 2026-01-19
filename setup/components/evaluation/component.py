"""Aurora evaluation Azure Machine Learning (AML) Component definition and deployment.

To create or update the component in your AML workspace, run:
    python -m setup.components.evaluation.component

By default, the workspace parameters (subscription_id, resource_group_name,
workspace_name) are retrieved from temporary environment variables defined in ./.env -
see ./.template.env for details on creating this file.
"""

from azure.ai.ml import Input
from azure.ai.ml.entities import CommandComponent

from setup.common.utils import create_mlclient

if __name__ == "__main__":
    component = CommandComponent(
        name="aurora_evaluation",
        display_name="Aurora Evaluation",
        description="Component for performing evaluation with Aurora.",
        version="1",
        command=(
            "python -m main "
            "--target ${{inputs.target}} "
            "--predictions ${{inputs.predictions}} "
            "--start_datetime ${{inputs.start_datetime}} "
            "--steps ${{inputs.steps}} "
        ),
        code="./setup/components/evaluation/",
        # environment gets persisted in registered component
        environment="aurora-environment:1",
        # mode and path are stripped from inputs and outputs on registration
        inputs={
            "data": Input(
                type="uri_folder",
                description="Data asset containing the ground truth state.",
            ),
            "prediction": Input(
                type="uri_folder",
                description="Data asset containing the predicted state.",
            ),
            "start_datetime": Input(
                type="string",
                description=(
                    "Start ISO 8601 format datetime for initial state data e.g. "
                    "2026-01-01T00:00:00."
                ),
            ),
            "steps": Input(
                type="integer",
                min=1,
                max=10,
                description="Number of steps in the future that were inferred.",
            ),
        },
        outputs={},
        instance_count=1,
        # whether to re-run the component when inputs are the same
        is_deterministic=True,
        # include common utils and constants for use in remote code
        additional_includes=["./setup/components/common/"],
    )
    mlc = create_mlclient(local=True)
    mlc.components.create_or_update(component)
