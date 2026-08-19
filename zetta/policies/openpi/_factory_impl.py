"""Lazy implementation derived from RLinf's OpenPI model loader."""

from zetta.policies.openpi.openpi_action_model import (  # noqa: F401
    OpenPi0Config,
    OpenPi0ForRLActionPrediction,
)


def get_model(cfg, torch_dtype=None):
    # The body is the upstream RLinf OpenPI loader, intentionally isolated from
    # the global RLinf model registry.  Importing dataconfig remains lazy.
    import glob
    import os
    import pathlib

    import openpi.shared.download as download
    import openpi.transforms as transforms
    import safetensors
    import torch
    from openpi.training import checkpoints as checkpoints_module

    from zetta.policies.openpi.dataconfig import get_openpi_config

    config_name = getattr(cfg.openpi, "config_name", None)
    data_kwargs = getattr(cfg, "openpi_data", None)
    train_config = get_openpi_config(config_name, model_path=cfg.model_path, data_kwargs=data_kwargs)
    model_config = OpenPi0Config(**train_config.model.__dict__)
    for key, value in getattr(cfg, "openpi", {}).items():
        model_config.__dict__[key] = value
    checkpoint_dir = download.maybe_download(str(cfg.model_path))
    model = OpenPi0ForRLActionPrediction(model_config)
    full_weights = os.path.join(checkpoint_dir, "model_state_dict", "full_weights.pt")
    actor_weights = os.path.join(checkpoint_dir, "actor", "model_state_dict", "full_weights.pt")
    if os.path.exists(full_weights):
        model.load_state_dict(torch.load(full_weights, map_location="cpu"), strict=False)
    elif os.path.exists(actor_weights):
        model.load_state_dict(torch.load(actor_weights, map_location="cpu"), strict=False)
    else:
        weights = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        weights = weights or [os.path.join(checkpoint_dir, "model.safetensors")]
        state = {}
        for path in weights:
            state.update(safetensors.torch.load_file(path, device="cpu"))
        model.load_state_dict(state, strict=False)
    if model_config.train_expert_only:
        model.freeze_vlm()
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    if bool(getattr(cfg, "load_to_device", True)):
        target_device = str(getattr(cfg, "device", "cuda") or "cuda")
        model = model.to(torch.device(target_device))
    data_config = train_config.data.create(train_config.assets_dirs, model_config)
    norm_stats_path = data_kwargs.get("norm_stats_path") if data_kwargs else None
    if norm_stats_path is not None:
        norm_stats = data_config.norm_stats
        if norm_stats is None:
            norm_dir = pathlib.Path(norm_stats_path).expanduser()
            if norm_dir.is_file():
                norm_dir = norm_dir.parent
            norm_stats = checkpoints_module.load_norm_stats(
                norm_dir.parent, norm_dir.name
            )
    else:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats")
        norm_stats = checkpoints_module.load_norm_stats(checkpoint_dir, data_config.asset_id)
    model.setup_wrappers(
        transforms=[
            transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
    )
    return model
