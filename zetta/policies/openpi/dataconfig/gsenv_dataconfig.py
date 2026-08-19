import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from zetta.policies.openpi.policies import gsenv_policy


@dataclasses.dataclass(frozen=True)
class LeRobotGSEnvDataConfig(DataConfigFactory):
    extra_delta_transform: bool = False

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.image",
                        "observation/state": "observation.state",
                        "actions": "actions",
                        "prompt": "prompt",  # 'task_descriptions'
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[gsenv_policy.GSEnvInputs(model_type=model_config.model_type)],
            outputs=[gsenv_policy.GSEnvOutputs()],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
