
import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from zetta.policies.openpi.policies import isaaclab_policy


@dataclasses.dataclass(frozen=True)
class LeRobotIsaacLabStackCubeDataConfig(DataConfigFactory):
    """OpenPI data config aligned with stack-cube fine-tuning recipe."""

    default_prompt: str | None = (
        "Stack the red block on the blue block, then stack the green block on the red block"
    )

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.front",
                        "observation/wrist_image": "observation.images.wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[isaaclab_policy.IsaacLabInputs(model_type=model_config.model_type)],
            outputs=[isaaclab_policy.IsaacLabOutputs()],
        )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )
