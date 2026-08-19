"""Goal-S task0：``rollout_runtime``（ray_channel 底座）Pi0.5 + Critic-Recovery 在线延迟。

与 main 侧 ``examples/libero_pi05_critic_recovery/run.py`` 逐语义对应的单臂驱动脚本：
同一个 task/seed/chunk_size/horizon 契约，测的是"已部署 Critic-Recovery 的在线推理
延迟"，不执行、不计时任何训练或优化过程（Cluster/Diagnose/Candidate/Shadow
Replay/Same-seed/Held-out）。

```text
Goal-S task0 / seed0 reset（critic_rules 随 reset 传入）
→ Pi0.5（每次执行 5 个动作，走 rollout_runtime 的 rlinf policy backend）
→ 已冻结 Critic（rlinf_env.py 的 _evaluate_critic，每个物理动作检查一次）
→ Critic 触发后交由 Recovery（本脚本固定用一次 Pi0.5 fallback 重新规划，
  不复刻 main 侧 promoted bundle 的 semantic_joint_interact primitive——那是
  legacy 运动学专属实现，rollout_runtime 侧没有对应的 Recovery primitive 执行层，
  详见文件末尾说明）
→ LIBERO 官方 success / truncation
```

用 ``asyncio.run()`` 直接驱动，不走 pytest：远程 Linux GPU 主机的
``pytest-asyncio==1.4.0`` + ``backports.asyncio.runner==1.2.0`` 组合在 Python 3.10 下对
异步测试有已知挂起缺陷（与本脚本、Critic-Recovery 改动无关，见任务记录），本脚本因此
完全不依赖 pytest 基础设施。

用法（在已激活 rollout_runtime 依赖的 venv 下）::

    python3 scripts/experiments/libero_critic_recovery_latency_v3.py \\
        --output runs/v3-critic-recovery/seed-0 \\
        --seed 0 \\
        --env-cuda-device 5 \\
        --rollout-cuda-device 4 \\
        --model-path /abs/path/to/checkpoint

上面是 in-process 模式：本进程自建一整套 runtime（``build_local_components``），只
适合单次延迟测量，量不出 DynamicSlotPool 的并发扩缩容行为——多个这样的进程各自建
一套池，互不共享，"并发"只是墙上时钟意义上的并行，不是同一个池里的排队/扩容。

验证并发扩缩容需要 served 模式：先用 dynamic-pool preset 起一个共享服务
（``rollout-runtime serve --config rollout_runtime/config/presets/
a100_libero_pi05_pro_dynamic.yaml --launch ray``），再让多个本脚本的独立进程打到
同一个 ``--runtime-url``（``--env-pool-size`` 固定为 1，只变
``--env-max-pool-size``——同一个池的并发调用者必须声明相同的 ``pool_size``，否则各
自建独立的池；``max_dynamic_pool_size`` 不进池 digest，因此可以不同，但同一批测试
里保持一致更利于结果解读）::

    python3 scripts/experiments/libero_critic_recovery_latency_v3.py \\
        --output runs/v3-critic-recovery/served/seed-0 \\
        --seed 0 \\
        --runtime-url http://127.0.0.1:8710 \\
        --env-pool-size 1 --env-max-pool-size 4

served 模式下不需要 ``--model-path``（模型已在服务端加载一次），也不应该再传
``--env-cuda-device``/``--rollout-cuda-device``（那两个是 in-process 模式独有的
"进程内多卡隔离"手段，对一个远端共享服务没有意义）。

输出 ``latency.json`` / ``result.json``，字段与 main 侧 README 的延迟字段表一一对应
（``online_inference_end_to_end`` / ``time_to_critic_trigger`` / ... ），便于同一份
分析脚本或人工核对跨支报告。
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import statistics
import sys
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

SUITE = "libero_goal_swap"
TASK_ID = 0
TASK_LANGUAGE = "Open the middle layer of the drawer"
DEFAULT_CONFIG = REPO_ROOT / "scripts" / "experiments" / "critic-recovery-v3.json"
ACTIONS_PER_CHUNK = 5
MAX_EPISODE_STEPS = 310
WAIT_STEPS = 10
DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)


def _now_ns() -> int:
    """当前单调时钟（纳秒），只用于本进程内的相对计时。

    Returns:
        单调时钟纳秒值。
    """
    return time.perf_counter_ns()


def _elapsed_ms(start_ns: int) -> float:
    """自 ``start_ns`` 起经过的毫秒数，四舍五入到 3 位小数。

    Args:
        start_ns: 起始时间（``_now_ns()`` 的返回值）。

    Returns:
        经过的毫秒数。
    """
    return round((_now_ns() - start_ns) / 1_000_000.0, 3)


def _json_default(value: Any) -> Any:
    """给 ``json.dumps`` 用的兜底序列化：numpy 标量/数组、``Path``。

    Args:
        value: 待序列化的值。

    Returns:
        JSON 兼容表示。

    Raises:
        TypeError: 值既不是 numpy 也不是 ``Path``。
    """
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    """把结构写成规范化 JSON（sort_keys，末尾换行）。

    Args:
        path: 目标文件路径。
        value: 待写入的结构。
    """
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


@dataclass
class LatencyTrace:
    """逐事件延迟记录（与 main 侧 ``run.py::LatencyTrace`` 同构）。

    Attributes:
        phase: 当前阶段名，供 ``add`` 的事件按阶段筛选。
        events: 累积的事件列表。
    """

    phase: str = "startup"
    events: list[dict[str, Any]] = field(default_factory=list)

    @asynccontextmanager
    async def in_phase(self, phase: str) -> AsyncIterator[None]:
        """临时切换阶段名，退出时恢复。

        Args:
            phase: 临时阶段名。

        Yields:
            无。
        """
        previous = self.phase
        self.phase = str(phase)
        try:
            yield
        finally:
            self.phase = previous

    def add(self, kind: str, elapsed_ms: float, **metadata: Any) -> None:
        """追加一条事件记录。

        Args:
            kind: 事件类型（如 ``"pi05_predict"`` / ``"critic_action_chunk"``）。
            elapsed_ms: 本事件耗时（毫秒）。
            **metadata: 附加字段。
        """
        self.events.append(
            {
                "event_index": len(self.events) + 1,
                "phase": self.phase,
                "kind": str(kind),
                "elapsed_ms": float(elapsed_ms),
                **metadata,
            }
        )

    def select(
        self, *, kind: str | None = None, phase: str | None = None
    ) -> list[dict[str, Any]]:
        """按类型/阶段筛选事件。

        Args:
            kind: 事件类型；``None`` 不筛选。
            phase: 阶段名；``None`` 不筛选。

        Returns:
            匹配的事件列表。
        """
        return [
            row
            for row in self.events
            if (kind is None or row["kind"] == kind)
            and (phase is None or row["phase"] == phase)
        ]


def _summary(values: list[float]) -> dict[str, float | int | None]:
    """把一组耗时汇总成 count/total/mean/p50/max（与 main 侧同构）。

    Args:
        values: 毫秒耗时列表。

    Returns:
        汇总统计；空列表时全部字段为 0/``None``。
    """
    if not values:
        return {
            "count": 0,
            "total_ms": 0.0,
            "mean_ms": None,
            "p50_ms": None,
            "max_ms": None,
        }
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "total_ms": round(sum(ordered), 3),
        "mean_ms": round(statistics.fmean(ordered), 3),
        "p50_ms": round(statistics.median(ordered), 3),
        "max_ms": round(max(ordered), 3),
    }


def _event_summary(
    trace: LatencyTrace, kind: str, phase: str | None = None
) -> dict[str, float | int | None]:
    """``_summary`` 的便捷包装：先按类型/阶段筛选再汇总。

    Args:
        trace: 延迟记录。
        kind: 事件类型。
        phase: 阶段名；``None`` 不筛选。

    Returns:
        汇总统计。
    """
    return _summary([row["elapsed_ms"] for row in trace.select(kind=kind, phase=phase)])


def _prepare_output(path: Path) -> Path:
    """校验并创建输出目录：必须为空或不存在，防止两次结果混在一起。

    Args:
        path: 期望的输出目录。

    Returns:
        已创建的绝对路径。

    Raises:
        FileExistsError: 目录已存在且非空。
    """
    resolved = path.expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {resolved}; choose a new directory"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _load_config(path: Path) -> dict[str, Any]:
    """加载并校验 ``critic_rules`` 部署配置（与 main 侧 ``critic-recovery.json`` 同构）。

    Args:
        path: 配置文件路径。

    Returns:
        已解析的配置字典。

    Raises:
        ValueError: 配置目标的 suite/task/language 与本脚本冻结的契约不符。
    """
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("suite") != SUITE or int(config.get("task_id", -1)) != TASK_ID:
        raise ValueError("Critic-Recovery config does not target Goal-S task0")
    if config.get("task_language") != TASK_LANGUAGE:
        raise ValueError("Critic-Recovery config has the wrong task language")
    if len(config.get("critic_rules", ())) != 1:
        raise ValueError("this harness requires exactly one Critic rule")
    return config


def _build_runtime_config(args: argparse.Namespace) -> Any:
    """从 ``a100_libero_pi05_pro`` preset 收窄成单 env-rank + 单 rollout-rank 的配置。

    与 8-rank 的正式吞吐 preset 相比只改三处：env/rollout 各收窄到 1 rank、
    ``env_config`` 换成 Goal-S task0 契约（``libero_goal_swap``/task0/310 步）、
    ``policy_config.model_path`` 换成命令行给的 checkpoint。chunk_size=5 与
    gateway/transport/admission/payload 原样保留，与正式吞吐 preset 用的是同一份
    契约，跨 preset 的延迟数字才可比。

    **刻意不写 ``cluster.component_placement``**：那是给 ``launch/ray_launch.py``
    的真正跨进程部署用的（``rollout`` 的 placement 字符串枚举的是硬件 rank，决定
    ``ray.remote`` 起几个独立进程）。本脚本走 ``launch/local.py::build_local_components``
    ——EnvWorker / RolloutWorker 对象全部留在**本进程**，``transport.kind=ray_channel``
    只是换了 Gateway↔Worker 之间的消息通道实现（真实 rlinf Channel actor），不会真的
    把 worker 摆到独立 Ray 进程里。GPU 分配因此不走 placement 声明，而是在进程启动前
    用 ``CUDA_VISIBLE_DEVICES``（本函数不设置，留给 ``main()`` 在 import 任何 CUDA
    相关模块之前设置一次）+ ``rollout_worker.device="cuda:0"``（可见列表里的第一张）
    + ``MUJOCO_EGL_DEVICE_ID``（libero 子进程的真实 EGL 渲染设备号，由 ``main()``
    通过 ``zetta.utils.egl.configure_egl_device`` 探测得到——EGL 设备枚举顺序不
    保证等于 CUDA ordinal 顺序，不能假设它等于可见列表里的相对 index）
    组合完成，与 main 侧两个独立子进程各自 ``CUDA_VISIBLE_DEVICES`` 的方式不同，但
    效果一致：Pi0.5 推理与 MuJoCo/EGL 渲染落在不同物理 GPU 上。

    Args:
        args: 已解析的命令行参数。

    Returns:
        单 rank 的 ``RuntimeConfig``。
    """
    from rollout_runtime.config.schema import load_config

    config = load_config("a100_libero_pi05_pro")
    config.env_config = {
        **config.env_config,
        "task_suite_name": SUITE,
        "task_id": TASK_ID,
        "max_episode_steps": int(args.max_episode_steps),
        # 打开逐步 observation：main 侧 run.py 全程录像（episode.mp4 /
        # episode_wrist.mp4），本臂原先没有对应产物。preset 里的
        # a100_libero_pi05_pro.yaml 已经为"正常 agent 回合"打开过这一项
        # （见该文件头部注释 3），这里对齐同一个开关，不是新引入的能力。
        # ``--no-video`` 时关掉，省掉逐步 observation 的编解码/带宽成本
        # （与 main 侧 ``--no-video`` 语义一致）。
        "return_all_frames": not args.no_video,
    }
    config.env_worker.num_ranks = 1
    config.env_worker.max_sessions_per_rank = 1
    config.rollout_worker.num_ranks = 1
    config.rollout_worker.device = "cuda:0"
    config.rollout_worker.policy_config = {
        **config.rollout_worker.policy_config,
        "model_path": str(args.model_path),
    }
    config.cluster.component_placement = {}
    return config


async def _drain_critic_proposals(
    step_result: Any, *, trace: LatencyTrace, phase: str
) -> list[dict[str, Any]]:
    """从一次 ``action_step``/``policy_step`` 的 ``StepResult.info`` 里取 Critic proposal。

    Args:
        step_result: ``StepResult``（已 ``unwrap`` 过的成功结果）。
        trace: 延迟记录（未使用，保留位置与调用点对称，方便未来扩展逐步事件）。
        phase: 当前阶段名（未使用，同上）。

    Returns:
        本次 chunk 内命中的 proposal 列表；未配置 Critic 时为空列表。
    """
    del trace, phase  # 占位：当前实现不需要，保留签名给未来扩展用
    return list(step_result.info.get("critic_proposals", ()))


def _collect_frames(
    step_result: Any,
    *,
    main_frames: list[np.ndarray],
    wrist_frames: list[np.ndarray],
) -> None:
    """把一次 ``action_step`` 结果里逐步 observation 的图像解码后追加进帧缓冲。

    ``return_all_frames=True`` 时 ``StepResult.per_step`` 是这次 chunk 内每个物理
    步骤的 ``PerStepRecord``（D7），每条记录自带一份 ``Observation``；不打开该项时
    ``per_step`` 为 ``None``，本函数直接跳过（不追加任何帧，不报错——与 main 侧
    ``--no-video`` 语义一致，只是本脚本目前总是打开）。

    图像以 ``PayloadRef``（inline PNG）形式挂在 ``Observation.main_image`` /
    ``wrist_image`` 上，用 ``payload.decode_image`` 解成 ``[H, W, C]`` uint8 数组，
    与 main 侧 ``imageio.mimwrite`` 期望的输入形状一致。

    Args:
        step_result: ``StepResult``（已 ``unwrap`` 过的成功结果）。
        main_frames: 主视角帧缓冲（原地追加）。
        wrist_frames: 腕部视角帧缓冲（原地追加）。
    """
    from rollout_runtime.core import payload as payload_module

    per_step = step_result.per_step
    if not per_step:
        return
    for record in per_step:
        obs = record.observation
        if obs is None:
            continue
        if obs.main_image is not None:
            main_frames.append(payload_module.decode_image(obs.main_image))
        if obs.wrist_image is not None:
            wrist_frames.append(payload_module.decode_image(obs.wrist_image))


def _save_episode_video(
    output: Path,
    *,
    main_frames: list[np.ndarray],
    wrist_frames: list[np.ndarray],
    fps: int,
) -> dict[str, Any] | None:
    """把累积的帧缓冲编码成 mp4（与 main 侧 ``stop_recording_and_save`` 同构）。

    帧缓冲为空（比如某次运行没能打开 ``return_all_frames``，或 episode 0 步）时
    不写任何文件、返回 ``None``，不抛错——录像是诊断产物，不应让缺帧变成硬失败。

    Args:
        output: 输出目录。
        main_frames: 主视角帧列表（可能为空）。
        wrist_frames: 腕部视角帧列表（可能为空）。
        fps: 编码帧率。

    Returns:
        写入的文件信息（相对路径 + 帧数），全部为空时是 ``None``。
    """
    import imageio.v2 as imageio

    video_info: dict[str, Any] = {}
    if main_frames:
        main_path = output / "episode.mp4"
        imageio.mimwrite(str(main_path), main_frames, fps=fps, codec="libx264")
        video_info["episode_mp4"] = {
            "path": main_path.name, "frame_count": len(main_frames),
        }
    if wrist_frames:
        wrist_path = output / "episode_wrist.mp4"
        imageio.mimwrite(str(wrist_path), wrist_frames, fps=fps, codec="libx264")
        video_info["episode_wrist_mp4"] = {
            "path": wrist_path.name, "frame_count": len(wrist_frames),
        }
    return video_info or None


async def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    """跑一次 Goal-S task0 的在线 Critic-Recovery 延迟测量。

    Args:
        args: 已解析的命令行参数。

    Returns:
        ``(result_dict, exit_code)``；``exit_code`` 0 表示官方 success，2 表示未 success
        （截断 / Critic 未触发 / Recovery 未达成 success，仍是有效的在线运行）。

    Raises:
        ValueError: 参数不合法。
        RuntimeError: session 创建/reset 失败，或 episode 未产出任何结果。
    """
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if args.max_actions < 1:
        raise ValueError("max-actions must be positive")
    if args.env_pool_size < 1:
        raise ValueError("env-pool-size must be positive")
    if (
        args.env_max_pool_size is not None
        and args.env_max_pool_size < args.env_pool_size
    ):
        raise ValueError("env-max-pool-size must be >= env-pool-size")

    from rollout_runtime.api.errors import RuntimeApiError
    from rollout_runtime.api.messages import (
        CreateSessionRequest,
        EnvSpecMsg,
        PolicyRequest,
        ResetSpec,
    )
    from rollout_runtime.api.result import Err, unwrap
    from rollout_runtime.core import payload as payload_module

    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    output = _prepare_output(args.output)
    trace = LatencyTrace()
    harness_started = _now_ns()

    startup_started = _now_ns()
    if args.runtime_url:
        # served 模式：多个本脚本的独立进程可以打到**同一个** `rollout-runtime
        # serve` 实例，才谈得上验证 DynamicSlotPool 的并发扩缩容——in-process
        # `build_local_components` 每个进程自建一整套 runtime，各自的池互不
        # 共享，量不出真正的并发排队/扩容行为（见 RemoteRuntimeClient 模块
        # docstring）。env_config 因此不能再从本地 preset 收窄出来：真实值由
        # 服务端已加载的 preset 决定，本脚本只管 env_spec 的池容量字段。
        from rollout_runtime.serve.client import RemoteRuntimeClient, RemoteRuntime

        client = RemoteRuntimeClient(args.runtime_url, token=args.runtime_token)
        runtime = RemoteRuntime(client)
        await runtime.start()
        env_config: dict[str, Any] = {
            "task_suite_name": SUITE,
            "task_id": TASK_ID,
            "max_episode_steps": int(args.max_episode_steps),
            "return_all_frames": not args.no_video,
            "libero_variant": "pro",
        }
        transport_kind = "served"
    else:
        from rollout_runtime.launch.local import build_local_components

        runtime_config = _build_runtime_config(args)
        runtime = build_local_components(runtime_config)
        await runtime.start()
        env_config = dict(runtime_config.env_config)
        transport_kind = runtime_config.transport.kind
    env_startup_ms = _elapsed_ms(startup_started)

    episode_result: dict[str, Any] | None = None
    session_id: Any = None
    try:
        env_spec = EnvSpecMsg(
            env_family="libero",
            env_config=env_config,
            pool_size=int(args.env_pool_size),
            max_dynamic_pool_size=args.env_max_pool_size,
        )
        create_started = _now_ns()
        created = await runtime.gateway.create_sessions(
            [
                CreateSessionRequest(
                    application_id="critic-recovery-latency-v3",
                    client_session_key=f"goal-s0-seed-{args.seed}",
                    env_spec=env_spec,
                    default_policy_id="pi05",
                    lease_seconds=1800.0,
                )
            ]
        )
        failures = [row.error for row in created if isinstance(row, Err)]
        if failures:
            raise RuntimeError(f"create_sessions failed: {failures}")
        session_id = unwrap(created[0]).session_id
        session_create_ms = _elapsed_ms(create_started)

        reset_started = _now_ns()
        reset_result = unwrap(
            (
                await runtime.gateway.reset(
                    [session_id],
                    ResetSpec(
                        task_id=TASK_ID,
                        seed=int(args.seed),
                        instruction=TASK_LANGUAGE,
                        options={
                            "critic_rules": config["critic_rules"],
                            "critic_interrupt_on_proposal": True,
                        },
                    ),
                )
            )[0]
        )
        env_reset_ms = _elapsed_ms(reset_started)
        if reset_result.observation is None:
            raise RuntimeError("reset returned no observation")
        if reset_result.observation.instruction != TASK_LANGUAGE:
            raise RuntimeError(
                f"task language mismatch: expected={TASK_LANGUAGE!r} "
                f"actual={reset_result.observation.instruction!r}"
            )

        settling_started = _now_ns()
        settling_proposals: list[dict[str, Any]] = []
        episode_steps = 0
        main_frames: list[np.ndarray] = []
        wrist_frames: list[np.ndarray] = []
        async with trace.in_phase("settling"):
            for _ in range(int(args.wait_steps)):
                block = np.tile(DUMMY_ACTION, (1, 1))
                step_result = unwrap(
                    (
                        await runtime.gateway.action_step(
                            [session_id], [payload_module.encode_array(block)]
                        )
                    )[0]
                )
                if not args.no_video:
                    _collect_frames(
                        step_result, main_frames=main_frames, wrist_frames=wrist_frames
                    )
                episode_steps += step_result.executed_horizon
                proposals = await _drain_critic_proposals(
                    step_result, trace=trace, phase="settling"
                )
                settling_proposals.extend(proposals)
                if step_result.terminated or step_result.truncated or proposals:
                    break
        settling_ms = _elapsed_ms(settling_started)

        inference_started = _now_ns()
        critic_proposals = list(settling_proposals)
        pi05_chunks: list[dict[str, Any]] = []
        pi05_start_step = episode_steps
        terminated = False
        truncated = False
        critic_trigger_step: int | None = (
            int(critic_proposals[0]["step_index"]) if critic_proposals else None
        )
        while (
            not critic_proposals
            and not terminated
            and not truncated
            and episode_steps < int(args.max_episode_steps)
        ):
            chunk_started = _now_ns()
            async with trace.in_phase("pi05_before_critic"):
                infer_started = _now_ns()
                infer_result = unwrap(
                    (
                        await runtime.gateway.policy_infer(
                            [session_id], PolicyRequest(policy_id="pi05")
                        )
                    )[0]
                )
                trace.add(
                    "pi05_predict",
                    _elapsed_ms(infer_started),
                    model_version=infer_result.model_version,
                )
                if infer_result.actions is None:
                    raise RuntimeError("policy_infer returned no actions")
                step_result = unwrap(
                    (
                        await runtime.gateway.action_step(
                            [session_id], [infer_result.actions]
                        )
                    )[0]
                )
                if not args.no_video:
                    _collect_frames(
                        step_result, main_frames=main_frames, wrist_frames=wrist_frames
                    )
                trace.add(
                    "critic_action_chunk",
                    0.0,
                    requested_actions=ACTIONS_PER_CHUNK,
                    executed_actions=step_result.executed_horizon,
                )
            proposals = await _drain_critic_proposals(
                step_result, trace=trace, phase="pi05_before_critic"
            )
            episode_steps += step_result.executed_horizon
            terminated = step_result.terminated
            truncated = step_result.truncated
            pi05_chunks.append(
                {
                    "chunk": len(pi05_chunks) + 1,
                    "wall_ms": _elapsed_ms(chunk_started),
                    "actions_executed": step_result.executed_horizon,
                    "critic_proposals": proposals,
                }
            )
            if proposals:
                critic_proposals = proposals
                critic_trigger_step = int(proposals[0]["step_index"])

        critic_trigger_ms = _elapsed_ms(inference_started) if critic_proposals else None
        recovery_start_step = episode_steps
        fallback_result: dict[str, Any] | None = None
        fallback_ms: float | None = None

        # 本臂的 Recovery 是单次权威任务 Pi0.5 fallback：main 侧 promoted bundle 的
        # semantic_joint_interact 是一段专属运动学 primitive（直接操纵 EEF pose 逼近
        # 语义关节几何），rollout_runtime 没有对应的 Recovery primitive 执行层——Critic
        # 之后"谁来执行 Recovery 动作"在 v1 architecture 里明确不下沉进 Runtime Core
        # （只有 Critic 评估在环境侧，Recovery 决策与执行留给上层 Agent/primitive）。
        # 因此本臂在 Critic 触发后用权威完整任务文本继续 Pi0.5，直到 success 或
        # truncation，测的是"Critic 检测 + 继续规划"这条本臂实际具备的路径，不假装
        # 复刻一个本侧没有执行层的 primitive。
        if critic_proposals and not terminated and not truncated:
            fallback_started = _now_ns()
            async with trace.in_phase("recovery_pi05_fallback"):
                while (
                    not terminated
                    and not truncated
                    and episode_steps < int(args.max_episode_steps)
                ):
                    infer_started = _now_ns()
                    infer_result = unwrap(
                        (
                            await runtime.gateway.policy_infer(
                                [session_id], PolicyRequest(policy_id="pi05")
                            )
                        )[0]
                    )
                    trace.add(
                        "pi05_predict",
                        _elapsed_ms(infer_started),
                        model_version=infer_result.model_version,
                    )
                    if infer_result.actions is None:
                        raise RuntimeError("policy_infer returned no actions")
                    step_result = unwrap(
                        (
                            await runtime.gateway.action_step(
                                [session_id], [infer_result.actions]
                            )
                        )[0]
                    )
                    if not args.no_video:
                        _collect_frames(
                            step_result, main_frames=main_frames, wrist_frames=wrist_frames
                        )
                    trace.add(
                        "critic_action_chunk",
                        0.0,
                        requested_actions=ACTIONS_PER_CHUNK,
                        executed_actions=step_result.executed_horizon,
                    )
                    episode_steps += step_result.executed_horizon
                    terminated = step_result.terminated
                    truncated = step_result.truncated
            fallback_ms = _elapsed_ms(fallback_started)
            fallback_result = {"steps_used": episode_steps - recovery_start_step}

        inference_end_to_end_ms = _elapsed_ms(inference_started)
        inference_end_step = episode_steps

        final_state_started = _now_ns()
        async with trace.in_phase("final_telemetry"):
            final_state_result = unwrap(
                (
                    await runtime.gateway.extension_call(
                        [session_id], "libero", "critic_state", {}
                    )
                )[0]
            )
        final_state_ms = _elapsed_ms(final_state_started)

        official_success = bool(final_state_result.get("privileged.task.success", False))
        episode_result = {
            "schema": "goal-s0-critic-recovery-latency-v3/v1",
            "task": {
                "suite": SUITE,
                "task_id": TASK_ID,
                "language": TASK_LANGUAGE,
                "seed": args.seed,
            },
            "critic_recovery": {
                "config_path": str(config_path),
                "critic_rule_id": config["critic_rules"][0]["rule_id"],
                "recovery_kind": "pi05_fallback_only",
            },
            "runtime": {
                "env_cuda_device": args.env_cuda_device if not args.runtime_url else None,
                "rollout_cuda_device": (
                    args.rollout_cuda_device if not args.runtime_url else None
                ),
                "transport_kind": transport_kind,
                "runtime_url": args.runtime_url,
                "env_pool_size": int(args.env_pool_size),
                "env_max_pool_size": args.env_max_pool_size,
                "wait_steps": args.wait_steps,
                "max_episode_steps": args.max_episode_steps,
                "actions_per_chunk": ACTIONS_PER_CHUNK,
                "model_path": str(args.model_path) if args.model_path else None,
            },
            "outcome": {
                "official_success": official_success,
                "episode_terminated": bool(terminated),
                "episode_truncated": bool(truncated),
                "critic_triggered": bool(critic_proposals),
                "critic_trigger_step": critic_trigger_step,
                "pi05_fallback_executed": fallback_result is not None,
                "final_stage": final_state_result.get("privileged.task.stage.name"),
            },
            "counts": {
                "settling_actions": pi05_start_step,
                "pi05_chunks_before_critic": len(pi05_chunks),
                "pi05_actions_before_critic": recovery_start_step - pi05_start_step,
                "recovery_actions": inference_end_step - recovery_start_step,
                "online_inference_actions": inference_end_step - pi05_start_step,
                "episode_actions": inference_end_step,
            },
            "latency_ms": {
                "online_inference_end_to_end": inference_end_to_end_ms,
                "time_to_critic_trigger": critic_trigger_ms,
                "pi05_fallback": fallback_ms,
                "settling_before_inference": settling_ms,
                "env_startup": env_startup_ms,
                "session_create": session_create_ms,
                "env_reset": env_reset_ms,
                "final_state_telemetry": final_state_ms,
                "harness_before_shutdown": _elapsed_ms(harness_started),
                "pi05_predict_before_critic": _event_summary(
                    trace, "pi05_predict", "pi05_before_critic"
                ),
                "critic_action_chunks_before_trigger": _event_summary(
                    trace, "critic_action_chunk", "pi05_before_critic"
                ),
                "fallback_pi05_predict": _event_summary(
                    trace, "pi05_predict", "recovery_pi05_fallback"
                ),
                "fallback_critic_action_chunks": _event_summary(
                    trace, "critic_action_chunk", "recovery_pi05_fallback"
                ),
            },
            "critic_proposals": critic_proposals,
            "pi05_chunks": pi05_chunks,
            "fallback_result": fallback_result,
            "rpc_trace": trace.events,
        }
    finally:
        close_started = _now_ns()
        try:
            # 必须先释放本次创建的 slot，再关连接：``EnvPool.release()`` 只把
            # slot 放回 ``warm_free_slots``，不会自动发生——不主动
            # ``close_sessions`` 的话，served 模式下这个 slot 会在共享服务的
            # 池里**永久占用**（``_reserved_slot_count`` 只增不减，除非显式
            # ``shrink_idle``），直接的后果是"跑几轮并发测试后，池莫名其妙就
            # 满了、后面的并发档位在应该成功的位置也报 QUOTA_EXCEEDED"——这正是
            # served 并发压测脚本必须做、而单发 in-process 调试脚本可以偷懒不
            # 做的一步（in-process 模式每次调用都是全新进程、全新池，不存在
            # 跨调用的池状态残留）。
            if session_id is not None:
                with suppress(BaseException):
                    await runtime.gateway.close_sessions([session_id])
            # served 模式：``gateway`` 是 ``RemoteRuntimeClient``，没有 ``stop()``
            # ——远端 runtime 是共享的，客户端绝不能把它停掉（见
            # ``RemoteRuntime`` 模块 docstring 的"三个刻意的设计"第 1 条）。只有
            # in-process 形态才需要（也才可以）停掉本进程私有的那一套 worker。
            if not args.runtime_url:
                await runtime.gateway.stop()
        finally:
            await runtime.aclose()
        env_shutdown_ms = _elapsed_ms(close_started)

    if episode_result is None:
        raise RuntimeError("episode did not produce a result")
    episode_result["latency_ms"]["env_shutdown"] = env_shutdown_ms

    video_save_started = _now_ns()
    video_info = None
    if not args.no_video:
        video_info = _save_episode_video(
            output, main_frames=main_frames, wrist_frames=wrist_frames,
            fps=args.video_fps,
        )
    episode_result["latency_ms"]["video_save_outside_online_inference"] = _elapsed_ms(
        video_save_started
    )
    episode_result["video"] = video_info

    episode_result["latency_ms"]["full_harness_wall"] = _elapsed_ms(harness_started)
    _write_json(output / "result.json", episode_result)
    _write_json(
        output / "latency.json",
        {
            "schema": episode_result["schema"],
            "task": episode_result["task"],
            "critic_recovery": episode_result["critic_recovery"],
            "outcome": episode_result["outcome"],
            "counts": episode_result["counts"],
            "latency_ms": episode_result["latency_ms"],
            "rpc_trace": episode_result["rpc_trace"],
        },
    )
    print(json.dumps(episode_result["outcome"], sort_keys=True))
    print(f"online inference: {episode_result['latency_ms']['online_inference_end_to_end']:.3f} ms")
    print(f"result: {output / 'result.json'}")
    return episode_result, 0 if episode_result["outcome"]["official_success"] else 2


def _parser() -> argparse.ArgumentParser:
    """构造命令行解析器。

    Returns:
        已配置好全部选项的解析器。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-cuda-device", type=int, default=5)
    parser.add_argument("--rollout-cuda-device", type=int, default=4)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "in-process 模式必填：本进程要加载的 pi0.5 checkpoint。served 模式"
            "（给了 --runtime-url）下忽略——模型已经在服务端进程里加载好，"
            "每个并发客户端不应该、也不需要再各自加载一份。"
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--runtime-url",
        default=None,
        help=(
            "已经跑起来的 `rollout-runtime serve` 地址（如 http://127.0.0.1:8710）。"
            "给了这个参数就走 served/HTTP 模式（RemoteRuntimeClient），本脚本不再"
            "自建 in-process runtime；多个并发调用的本脚本进程可以打到同一个"
            "服务实例，才谈得上验证 DynamicSlotPool 的并发扩缩容/排队行为。"
            "不给则保持原有 in-process 行为（build_local_components，向后兼容）。"
        ),
    )
    parser.add_argument(
        "--runtime-token", default=None, help="served 模式下的 Bearer token（可选）"
    )
    parser.add_argument(
        "--env-pool-size",
        type=int,
        default=1,
        help=(
            "EnvSpecMsg.pool_size：进这个池的每个并发调用者都必须声明同一个值"
            "（它进 pool digest，见 api/messages.py），否则会各自建一个独立的池。"
            "并发扫描时固定为 1，只改 --env-max-pool-size（见下）。"
        ),
    )
    parser.add_argument(
        "--env-max-pool-size",
        type=int,
        default=None,
        help=(
            "EnvSpecMsg.max_dynamic_pool_size：DynamicSlotPool 的扩容上限。"
            "不进 pool digest，因此同一个池的并发调用者即使各自传不同的值也不冲突"
            "（EnvPool 只认第一个建池请求声明的上限，见 EnvPool._reserve_cold_"
            "create_locked 与 README 里记录的既有调度行为）。默认 None 表示不声明"
            "（等价于固定池，池满即 QUOTA_EXCEEDED，不会动态扩容）。"
        ),
    )
    parser.add_argument("--max-actions", type=int, default=300)
    parser.add_argument("--wait-steps", type=int, default=WAIT_STEPS)
    parser.add_argument("--max-episode-steps", type=int, default=MAX_EPISODE_STEPS)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--ray-tmp-dir",
        type=Path,
        default=None,
        help="可选的 Ray 会话/对象存储临时目录（RAY_TMPDIR）",
    )
    return parser


def main() -> int:
    """CLI 入口。

    在 import 任何 CUDA 相关模块（``torch`` / ``rlinf`` / ``robosuite``）之前设置
    ``CUDA_VISIBLE_DEVICES``：``build_local_components`` 是单进程架构，env/rollout
    的 worker 对象都在本进程里，物理 GPU 隔离因此只能靠"进程启动前只暴露两张目标卡，
    再用相对索引/独立的 EGL 设备 id 各自指到其中一张"完成（见
    ``_build_runtime_config`` 的说明），不是 ``cluster.component_placement``。
    ``MUJOCO_EGL_DEVICE_ID`` 必须用真实全局 CUDA ordinal 通过
    ``zetta.utils.egl.configure_egl_device`` 探测（EGL 设备枚举顺序不保证等于
    CUDA ordinal 顺序），且这一步必须在设置 ``CUDA_VISIBLE_DEVICES`` 之前完成——
    设置之后本进程看到的就是收窄过的相对 ordinal 空间，查不到真实映射。

    同样在 import ``ray`` 之前设置 ``RAY_TMPDIR``：``rlinf.scheduler.cluster.Cluster``
    内部固定构造 ``ray.init(...)`` 的参数，不读取任何自定义临时目录配置，也不接受
    本脚本传参（它是 third_party 代码，不应修改）。Ray 确认会读取 ``RAY_TMPDIR``
    环境变量作为会话/对象存储根目录（已验证：设置该变量后 ``session_dir`` 落在指定
    路径下，不需要显式传 ``_temp_dir``），因此在这里设置即可让 Ray 完全不触碰根分区。

    Returns:
        进程退出码（0 成功，2 未 success 但仍是有效在线运行）。

    Raises:
        ValueError: in-process 模式（没给 ``--runtime-url``）却没给 ``--model-path``
            ——served 模式下模型已经在服务端加载，这里才允许省略。
    """
    args = _parser().parse_args()
    if not args.runtime_url and args.model_path is None:
        raise ValueError(
            "--model-path is required unless --runtime-url is given "
            "(served mode loads the model server-side, once)"
        )
    if args.runtime_url:
        # served 模式：本进程只是个 HTTP 客户端，既不加载 pi0.5 也不起本地
        # MuJoCo/EGL 渲染，因此不需要、也不应该动 CUDA_VISIBLE_DEVICES / EGL /
        # RAY_TMPDIR ——那些都是 in-process 单进程架构专属的隔离手段（见下方
        # in-process 分支的说明），served 模式下服务端进程早已按自己的 preset
        # 设置好这些环境变量，客户端进程动它们没有意义，还可能在同一台机器上
        # 与并发跑的其它客户端互相冲突（多个客户端进程各自 mkdir 同一个
        # ray_tmp_dir 是安全的，但没必要）。
        _, exit_code = asyncio.run(run(args))
        return exit_code

    if args.ray_tmp_dir is not None:
        args.ray_tmp_dir.mkdir(parents=True, exist_ok=True)
        os.environ["RAY_TMPDIR"] = str(args.ray_tmp_dir)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    # EGL 设备顺序不保证等于 CUDA ordinal 顺序（同一坑 main 侧
    # ``robots/libero/env_server.py`` 已经踩过并注释过）：``zetta.utils.egl`` 用
    # ``EGL_NV_device_cuda`` 反查"真实 CUDA ordinal -> 真实 EGL device index"，
    # 必须用 ``env_cuda_device`` 这个真实全局编号查，且必须在设置
    # ``CUDA_VISIBLE_DEVICES``（会改变本进程能看到的 ordinal 空间）之前查。
    # 之前误把"可见列表里的相对 index"当作 EGL device id 硬编码成 ``"1"``，在
    # EGL 顺序与 CUDA 顺序不一致的机器上会被 robosuite 的
    # ``MUJOCO_EGL_DEVICE_ID in CUDA_VISIBLE_DEVICES`` 断言直接拒绝（子串检查，
    # 不是数值检查），此前几次“间歇性死锁”实为该断言在某些 GPU 组合下偶然不触发。
    from zetta.utils.egl import configure_egl_device

    configure_egl_device(args.env_cuda_device)
    os.environ["CUDA_VISIBLE_DEVICES"] = (
        f"{args.rollout_cuda_device},{args.env_cuda_device}"
    )
    # rollout_worker.device="cuda:0" 因此落到可见列表第一张（rollout_cuda_device）。
    os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")
    os.environ.setdefault("LIBERO_TYPE", "pro")
    _, exit_code = asyncio.run(run(args))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
