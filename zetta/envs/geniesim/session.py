# Copyright (c) 2026 Zetta Contributors
"""Native single-task calls, used only inside the Isaac interpreter.

The caller runs this object on a task thread while the main thread services
APICore's physics/render queues, as in the pinned upstream benchmark.
"""

from __future__ import annotations

import json
import random
import time
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .config import ACTION_DIM, CAMERA_NAMES, MAX_CHUNK, GenieSimConfig, flatten_state


def plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return plain(value.value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def validate_actions(actions: Any, limits: Any) -> np.ndarray:
    raw = np.asarray(actions)
    if raw.dtype.kind not in "fiu":
        raise ValueError("actions must contain real numbers")
    block = raw.astype(np.float64)
    if (
        block.ndim != 2
        or block.shape[1] != ACTION_DIM
        or not 1 <= len(block) <= MAX_CHUNK
    ):
        raise ValueError(f"actions must have shape [1..{MAX_CHUNK}, {ACTION_DIM}]")
    if not np.isfinite(block).all():
        raise ValueError("actions must be finite")
    bounds = np.asarray(limits, dtype=np.float64)
    if bounds.shape != (ACTION_DIM, 2) or not np.isfinite(bounds).all():
        raise ValueError("missing finite native joint limits")
    if np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("invalid native joint limit interval")
    if np.any(block < bounds[:, 0]) or np.any(block > bounds[:, 1]):
        raise ValueError("absolute joint target is outside native joint limits")
    return block


def action_limits_from_articulation(
    articulation: Any, names: list[str]
) -> list[list[float]]:
    properties = articulation.dof_properties
    indices = {name: index for index, name in enumerate(articulation.dof_names)}
    selected = properties[[indices[name] for name in names]]
    if (
        len(names) != ACTION_DIM
        or not np.all(selected["hasLimits"])
        # PhysX DofType.Rotation is 0 in Isaac 5.1. SingleArticulation's
        # property docstring still lists the older dynamic-control enum.
        or not np.all(selected["type"] == 0)
    ):
        raise RuntimeError(
            "expected 16 bounded rotational joints in the native G2 articulation"
        )
    limits = np.column_stack((selected["lower"], selected["upper"]))
    validate_actions(limits.mean(axis=1)[None, :], limits)
    return limits.tolist()


class GenieSimSession:
    def __init__(self, config: GenieSimConfig, cfg: Any, api: Any, directory: Path):
        self.config, self.cfg, self.api, self.directory = config, cfg, api, directory
        self.env = None
        self.seed = None
        self.episode = 0
        self.finished = False
        self.success = False
        self.evaluation_step = 0
        self.observation = None
        self.score = None
        self.limits = None

    def _create(self, seed: int) -> None:
        from geniesim_benchmark.benchmark.task_benchmark import TaskBenchmark
        from geniesim_benchmark.plugins.tgs import TaskGenerator
        from geniesim_benchmark.utils import system_utils
        from geniesim_benchmark.utils.generalization_utils import update_init_env
        from geniesim_benchmark.utils.name_utils import robot_type_mapping

        self.cfg.benchmark.seed = seed
        self.cfg.layout.seed = seed
        self.benchmark = TaskBenchmark(self.cfg.benchmark, self.api)
        task_config = system_utils.load_json(
            str(
                Path(system_utils.benchmark_conf_path())
                / "eval_tasks/table_task_1_g2_op.json"
            )
        )
        task_config.update(
            specific_task_name="table_task_1_g2_op",
            sub_task_name="pick_block_color",
            instruction_mode="full",
            language_perturbation=False,
        )
        task_config["scene"].update(
            scene_instance_id=0,
            sub_usd_override_root=str(
                Path(self.config.assets_root) / "zetta_tasks/pick_block_color"
            ),
        )
        generator = TaskGenerator(task_config)
        generated = self.directory / "generated_tasks"
        generator.generate_tasks(
            save_path=str(generated),
            task_name=task_config["task"],
            gen_config=task_config.get("generalization", {}),
        )
        task_config["robot"]["robot_init_pose"].update(generator.robot_init_pose)
        task_config["robot_cfg"] = robot_type_mapping("G2_omnipicker")
        self.benchmark.task_config = task_config
        self.benchmark.data_courier.set_robot_cfg(task_config["robot_cfg"])
        episode_file = sorted(generated.glob("*.json"))[0]
        self.benchmark.create_env(str(episode_file), 0)
        self.api.collect_init_physics()
        self.env = self.benchmark.env
        update_init_env(
            self.env, task_config, system_utils.load_json(str(episode_file))
        )
        self.env.apply_generalization(self.api, task_config)
        self.env.set_infer_status(True)
        self.env.set_depth_status(False)
        self.env.settle_due = True
        self.seed = seed

    def reset(self, seed: int) -> dict[str, Any]:
        from geniesim_benchmark.plugins.output_system import TaskEvaluation

        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if self.seed is not None and seed != self.seed:
            raise ValueError("a different seed requires a fresh simulator process")
        random.seed(seed)
        np.random.seed(seed)
        if self.env is None:
            self._create(seed)
        else:
            self.env.cancel_action("eval")
        self.episode += 1
        self.finished = self.success = False
        self.evaluation_step = 0
        self.env.current_step = 0
        self.env.set_current_task(0)
        self.env.reset()
        raw = self._settle_reset()
        # LLMTask resolves problem_name during reset, after parsing its checkers.
        if self.env.task.problem_name != "pick_block_color":
            raise RuntimeError("upstream task selection fell back to another task")
        self.score = TaskEvaluation(
            task_name="table_task_1_g2_op", sub_task_name="pick_block_color"
        )
        self.score.summarize_scores()
        self.env.do_eval_action()
        names = self.env.cfg["arm_joints"] + self.env.cfg["gripper_joints"]
        self.limits = self.api.run_on_physics_loop(
            lambda: action_limits_from_articulation(self.api._get_articulation(), names)
        )
        self.observation = self._snapshot(raw)
        self._save_result()
        return {
            "observation": self.observation,
            "action_limits": self.limits,
            "result": self.result(),
        }

    def _settle_reset(self) -> dict[str, Any]:
        # PiEnv.reset snaps positions but leaves prior drive targets active and
        # stops polling after one second. Synchronize the native reset targets.
        targets = {}
        items = []
        for positions, names in (
            (self.env.init_arm, self.env.cfg["arm_joints"]),
            (self.env.init_waist, self.env.cfg["waist_joints"]),
            (self.env.init_head, self.env.cfg["head_joints"]),
            (self.env.cfg["init_gripper_open"], self.env.cfg["gripper_joints"]),
        ):
            targets.update(zip(names, positions, strict=True))
            if names:
                items.append(
                    (
                        list(positions),
                        [self.env.robot_joint_indices[name] for name in names],
                        True,
                    )
                )
        self.api.set_joint_positions_batched(items)
        deadline = time.monotonic() + 30.0
        while True:
            joints = self.env.data_courier.get_joint_state_dict()
            if joints and all(
                abs(joints[name] - value) < 0.01 for name, value in targets.items()
            ):
                return self.env.get_observation()
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "native reset joints did not settle within 30 seconds"
                )
            time.sleep(0.02)

    def _snapshot(self, raw: dict[str, Any]) -> dict[str, Any]:
        states = plain(raw["states"])
        flatten_state(states)
        images = {}
        for name in CAMERA_NAMES:
            pixels = np.asarray(raw["images"][name])
            if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
                raise RuntimeError(f"invalid {name} RGB observation")
            images[name] = {"shape": list(pixels.shape), "data": pixels.tobytes()}
        instruction = self.env.task.get_instruction()
        if isinstance(instruction, (list, tuple)):
            instruction = instruction[0]
        return {
            "images": images,
            "states": states,
            "instruction": str(instruction),
            "step_index": int(self.env.current_step),
            "seed": self.seed,
            "episode": self.episode,
            "official_success": self.success,
            "evaluation_step": self.evaluation_step,
            "official_result": plain(self.score.result),
        }

    def step(self, actions: Any) -> dict[str, Any]:
        if self.observation is None or self.finished:
            raise RuntimeError("reset is required before stepping this episode")
        block = validate_actions(actions, self.limits)
        records = []
        for action in block:
            raw, terminated, updated, progress = self.env.step(
                {
                    "arm": action[:14].tolist(),
                    "gripper": action[14:].tolist(),
                }
            )
            if updated:
                self.score.update_progress(progress)
                self.score.summarize_scores()
                self.evaluation_step = int(self.env.current_step)
            success = self.score.result.get("scores", {}).get("E2E") == 1
            reward = float(success and not self.success)
            self.success = self.success or success
            truncated = (
                not terminated and self.env.current_step >= self.config.max_steps
            )
            self.finished = bool(terminated or truncated)
            self.observation = self._snapshot(raw)
            record = {
                "step_index": int(self.env.current_step),
                "states": self.observation["states"],
                "action": action.tolist(),
                "reward": reward,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "evaluation_updated": bool(updated),
                "evaluation_step": self.evaluation_step,
                "official_success": self.success,
            }
            records.append(record)
            with (self.directory / f"episode-{self.episode:04d}.jsonl").open(
                "a"
            ) as stream:
                stream.write(json.dumps(record) + "\n")
            self.benchmark.data_courier.sleep()
            if self.finished:
                self.env.cancel_action("eval")
                break
        self._save_result()
        return {
            "observation": self.observation,
            "steps": records,
            "result": self.result(),
        }

    def result(self) -> dict[str, Any]:
        if self.observation is None:
            raise RuntimeError("reset is required before reading episode results")
        terminated = bool(self.env.has_done)
        truncated = bool(self.finished and not terminated)
        return {
            "task": self.config.task,
            "seed": self.seed,
            "episode": self.episode,
            "control_steps": int(self.env.current_step),
            "evaluation_step": self.evaluation_step,
            "terminated": terminated,
            "truncated": truncated,
            "success": self.success,
            "reason": ("success" if self.success else "task_failure")
            if terminated
            else "time_limit"
            if truncated
            else "running",
            "official_result": plain(self.score.result),
        }

    def _save_result(self) -> None:
        (self.directory / f"episode-{self.episode:04d}.json").write_text(
            json.dumps(self.result(), indent=2) + "\n"
        )

    def close(self) -> None:
        if self.env is not None:
            if self.observation is not None:
                self._save_result()
            self.env.stop()
