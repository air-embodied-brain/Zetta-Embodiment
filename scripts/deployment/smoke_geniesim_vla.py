#!/usr/bin/env python3
# Copyright (c) 2026 Zetta Contributors
"""Run paired Genie Sim episodes with a real policy or an explicit protocol stub.

The stub holds the observed joints. It validates native simulator / Runtime /
WebSocket / Campaign integration and never claims pretrained-model quality.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from robots.geniesim.run_rollout import build_parser as rollout_parser  # noqa: E402
from robots.geniesim.run_rollout import run  # noqa: E402
from rollout_runtime.config.schema import load_config  # noqa: E402
from rollout_runtime.launch.local import build_local_components  # noqa: E402
from zetta.envs.geniesim.config import GenieSimConfig  # noqa: E402
from zetta.evolution.gating import evaluate_paired_gate  # noqa: E402
from zetta.evolution.jsonio import atomic_write_json, read_json  # noqa: E402
from zetta.evolution.models import (  # noqa: E402
    CandidateBundle,
    CriticRule,
    RecoveryRule,
    RecoveryStep,
)


@contextmanager
def policy_service(args):
    if not args.protocol_stub:
        if not args.model_version:
            raise ValueError("--policy-endpoint requires --model-version")
        yield args.policy_endpoint, args.model_version
        return
    import msgpack
    from websockets.sync.server import serve

    version = "protocol-stub-hold-v1"

    def handle(socket):
        for packet in socket:
            request = msgpack.unpackb(packet, raw=False)["params"]
            joints = request["states"]
            arms, grippers = joints["arm_joint_states"], joints["gripper_states"]
            result = {
                "left_arm": {"kind": "JOINT_ABS", "values": [arms[:7]] * 5},
                "right_arm": {"kind": "JOINT_ABS", "values": [arms[7:]] * 5},
                "left_effector": [[grippers[0]]] * 5,
                "right_effector": [[grippers[1]]] * 5,
                "model_version": version,
                "policy_rng": request["policy_rng"],
            }
            socket.send(msgpack.packb({"result": result}, use_bin_type=True))

    with serve(handle, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"ws://127.0.0.1:{server.socket.getsockname()[1]}", version
        finally:
            server.shutdown()
            thread.join(timeout=5)


def smoke_bundle():
    return CandidateBundle(
        candidate_id="geniesim-protocol-hold-smoke",
        generation=1,
        parent_sha256=None,
        diagnosis_sha256="0" * 64,
        causal_hypothesis="Protocol smoke: exercise the frozen recovery path",
        mechanism_change="Hold initial joints for two control steps",
        validation_plan="Compare same-seed reset identity and audit executed recovery",
        critic_rules=(
            CriticRule(
                rule_id="initial",
                title="Initial boundary",
                feature="episode.step",
                operator="eq",
                threshold=0,
                dwell_steps=1,
                cooldown_steps=64,
                proposal="Exercise a hold recovery",
                evidence_ids=("protocol-smoke",),
            ),
        ),
        recovery_rules=(
            RecoveryRule(
                recovery_id="hold",
                title="Hold smoke",
                trigger_rule_ids=("initial",),
                precondition="At reset",
                steps=(
                    RecoveryStep(
                        tool="geniesim.hold",
                        parameters={"max_steps": 2},
                        stop_when="budget_exhausted_or_success",
                    ),
                ),
                safety_constraints=("Native limits",),
                stop_condition="official_success_or_budget",
                fallback="resume_vla",
                evidence_ids=("protocol-smoke",),
            ),
        ),
    )


async def smoke(args, endpoint, version):
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    native = GenieSimConfig.from_mapping(read_json(args.env_config))
    native = dataclasses.replace(native, output_root=str(root / "native"))
    config_path = root / "env-config.json"
    atomic_write_json(config_path, native.to_dict(), overwrite=False)
    preset = load_config("geniesim_vla")
    preset = dataclasses.replace(
        preset,
        env_config=native.to_dict(),
        rollout_worker=dataclasses.replace(
            preset.rollout_worker,
            policy_config={
                "endpoint": endpoint,
                "model_version": version,
                "actions_per_chunk": 5,
            },
        ),
    )
    bundle = smoke_bundle()
    bundle_path = root / "candidate.json"
    atomic_write_json(bundle_path, bundle.as_dict(), overwrite=False)
    records = []
    async with build_local_components(preset) as runtime:
        for arm in ("parent", "candidate"):
            options = rollout_parser().parse_args(
                [
                    "--runtime-url",
                    "http://127.0.0.1:1",
                    "--env-config",
                    str(config_path),
                    "--policy-model-version",
                    version,
                    "--seed",
                    str(args.seed),
                    "--policy-rng",
                    "20260911",
                    "--logical-id",
                    arm,
                    "--output-dir",
                    str(root / arm),
                    "--capture-frames",
                ]
            )
            if arm == "candidate":
                options.bundle, options.bundle_sha256 = str(bundle_path), bundle.sha256
                options.generation = 1
            record = await run(options, client=runtime.gateway)
            records.append(record)
            if record.status != "valid":
                raise RuntimeError(record.invalid_reason)
    processes = [read_json(path) for path in (root / "native").glob("*/process.json")]
    if not processes or any(
        row.get("status") != "closed" or row.get("returncode") != 0 for row in processes
    ):
        raise RuntimeError("native process did not acknowledge clean shutdown")
    decision = evaluate_paired_gate(
        kind="same_seed",
        candidate_sha256=bundle.sha256,
        parent_sha256=None,
        candidate_records=[records[1]],
        parent_records=[records[0]],
        expected_seeds=(args.seed,),
    )
    if not records[1].artifact_index["candidate_intervention"]:
        raise RuntimeError("candidate recovery never executed")
    report = {
        "status": "passed",
        "policy_kind": "protocol_stub" if args.protocol_stub else "external_vla",
        "model_version": version,
        "pretrained_model_validated": not args.protocol_stub,
        "episodes": [record.as_dict() for record in records],
        "paired_gate": decision.as_dict(),
        "native_processes": processes,
    }
    atomic_write_json(root / "result.json", report, overwrite=False)
    print(f"passed: {root / 'result.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1001)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--policy-endpoint")
    group.add_argument("--protocol-stub", action="store_true")
    parser.add_argument("--model-version")
    args = parser.parse_args()
    with policy_service(args) as (endpoint, version):
        asyncio.run(smoke(args, endpoint, version))


if __name__ == "__main__":
    main()
