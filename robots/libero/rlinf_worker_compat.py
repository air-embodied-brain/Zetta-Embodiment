# Copyright (c) 2026 RPent Contributors
"""Compatibility bridge for RLinf LIBERO workers without ``env_call``.

RPent's privileged evidence lives inside the spawned LIBERO environment. Some
RLinf releases expose camera-specific worker commands but no generic method
call, so the parent process cannot read that evidence. The installer below is
idempotent and leaves newer, native implementations untouched.
"""

from __future__ import annotations

from multiprocessing import connection
from typing import Any


def _resolve_target(environment: Any, target: str) -> Any:
    if target in {"", "self"}:
        return environment
    if target == "unwrapped":
        return environment.unwrapped
    value = environment
    for component in target.split("."):
        if not component or component == "self":
            continue
        value = getattr(value, component)
    return value


def _dispatch_env_call(environment: Any, payload: dict[str, Any]) -> Any:
    method = str(payload["method"])
    args = payload.get("args", ())
    kwargs = payload.get("kwargs", {})
    target = str(payload.get("target", "self"))
    if not isinstance(args, (list, tuple)):
        raise TypeError("env_call args must be a list or tuple")
    if not isinstance(kwargs, dict):
        raise TypeError("env_call kwargs must be an object")
    owner = _resolve_target(environment, target)
    return getattr(owner, method)(*args, **kwargs)


def _parent_env_call(
    worker: Any,
    method: str,
    *,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    target: str = "self",
) -> Any:
    worker.parent_remote.send(
        [
            "env_call",
            {
                "method": str(method),
                "args": tuple(args),
                "kwargs": dict(kwargs or {}),
                "target": str(target),
            },
        ]
    )
    return worker.parent_remote.recv()


def compat_worker(
    parent: connection.Connection,
    pipe: connection.Connection,
    env_fn_wrapper: Any,
    obs_bufs: Any = None,
) -> None:
    """RLinf's LIBERO worker loop plus the RPent ``env_call`` command."""

    from rlinf.envs.libero import venv as upstream
    from rlinf.envs.venv import ShArray

    def encode_obs(observation: Any, buffer: Any) -> None:
        import numpy as np

        if isinstance(observation, np.ndarray) and isinstance(buffer, ShArray):
            buffer.save(observation)
        elif isinstance(observation, tuple) and isinstance(buffer, tuple):
            for value, nested in zip(observation, buffer):
                encode_obs(value, nested)
        elif isinstance(observation, dict) and isinstance(buffer, dict):
            for key, value in observation.items():
                encode_obs(value, buffer[key])

    parent.close()
    environment = env_fn_wrapper.data()
    try:
        while True:
            try:
                command, data = pipe.recv()
            except EOFError:
                pipe.close()
                break
            if command == "step":
                result = environment.step(data)
                if obs_bufs is not None:
                    encode_obs(result[0], obs_bufs)
                    result = (None, *result[1:])
                pipe.send(result)
            elif command == "reset":
                result = environment.reset(**data)
                returns_info = (
                    isinstance(result, (tuple, list))
                    and len(result) == 2
                    and isinstance(result[1], dict)
                )
                if returns_info:
                    observation, info = result
                else:
                    observation = result
                if obs_bufs is not None:
                    encode_obs(observation, obs_bufs)
                    observation = None
                pipe.send((observation, info) if returns_info else observation)
            elif command == "close":
                pipe.send(environment.close())
                pipe.close()
                break
            elif command == "render":
                pipe.send(
                    environment.render(**data)
                    if hasattr(environment, "render")
                    else None
                )
            elif command == "seed":
                if hasattr(environment, "seed"):
                    pipe.send(environment.seed(data))
                else:
                    environment.reset(seed=data)
                    pipe.send(None)
            elif command == "getattr":
                pipe.send(
                    getattr(environment, data) if hasattr(environment, data) else None
                )
            elif command == "setattr":
                setattr(environment.unwrapped, data["key"], data["value"])
            elif command == "check_success":
                pipe.send(environment.check_success())
            elif command == "get_segmentation_of_interest":
                pipe.send(environment.get_segmentation_of_interest(data))
            elif command == "get_sim_state":
                pipe.send(environment.get_sim_state())
            elif command == "set_init_state":
                observation = environment.set_init_state(data)
                pipe.send(observation)
            elif command == "reconfigure":
                environment.close()
                seed = data.pop("seed")
                environment = upstream.OffScreenRenderEnv(**data)
                environment.seed(seed)
                pipe.send(None)
            elif command == "get_camera_meta":
                from robosuite.utils import camera_utils

                robosuite_env = getattr(environment, "env", environment)
                while hasattr(robosuite_env, "env"):
                    robosuite_env = robosuite_env.env
                simulation = robosuite_env.sim
                camera = data.get("camera_name", "agentview")
                height = int(data.get("height", 256))
                width = int(data.get("width", 256))
                intrinsic = camera_utils.get_camera_intrinsic_matrix(
                    simulation, camera, height, width
                )
                extrinsic = camera_utils.get_camera_extrinsic_matrix(
                    simulation, camera
                )
                extent = float(simulation.model.stat.extent)
                pipe.send(
                    {
                        "camera_name": camera,
                        "height": height,
                        "width": width,
                        "intrinsic_K": intrinsic.tolist(),
                        "extrinsic_cam2world": extrinsic.tolist(),
                        "depth_near": float(simulation.model.vis.map.znear) * extent,
                        "depth_far": float(simulation.model.vis.map.zfar) * extent,
                    }
                )
            elif command == "render_camera":
                robosuite_env = getattr(environment, "env", environment)
                while hasattr(robosuite_env, "env"):
                    robosuite_env = robosuite_env.env
                simulation = robosuite_env.sim
                pipe.send(
                    simulation.render(
                        width=int(data.get("width", 1024)),
                        height=int(data.get("height", 1024)),
                        camera_name=data.get("camera_name", "agentview"),
                        depth=bool(data.get("depth", False)),
                    )
                )
            elif command == "env_call":
                pipe.send(_dispatch_env_call(environment, data))
            else:
                pipe.close()
                raise NotImplementedError(command)
    except KeyboardInterrupt:
        pipe.close()


def install_rlinf_env_call_compat() -> str:
    """Install the bridge for the active RLinf module when it is needed."""

    from rlinf.envs.libero import venv

    worker_type = venv.ReconfigureSubprocEnvWorker
    if hasattr(worker_type, "env_call"):
        return "native"
    worker_type.env_call = _parent_env_call
    venv._worker = compat_worker
    return "installed"
