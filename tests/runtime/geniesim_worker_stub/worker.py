# Copyright (c) 2026 Zetta Contributors
"""A real child process for supervision and Runtime transport regression tests."""

import argparse
import json
import os
import socket
import time
from pathlib import Path

from .wire import PROTOCOL_VERSION, receive_packet, send_packet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--connection-fd", type=int)
    parser.add_argument("--run-directory", type=Path)
    args = parser.parse_args()
    channel = socket.socket(fileno=args.connection_fd)
    hello = receive_packet(channel, time.monotonic() + 10)
    config = hello["config"]
    mode = os.environ.get("ZETTA_GENIESIM_TEST_MODE", "normal")
    if mode == "startup_exit":
        raise SystemExit(12)
    if mode == "startup_timeout":
        time.sleep(60)
    send_packet(
        channel,
        {
            "kind": "ready",
            "protocol": PROTOCOL_VERSION,
            "pid": os.getpid(),
        },
        time.monotonic() + 10,
    )
    step, seed, episode = 0, 0, 0
    state = [0.2] * 14 + [0.7, 0.7]

    def observation():
        groups = {
            "right_arm": state[7:14],
            "waist": [0.0] * 5,
            "head": [],
            "left_gripper": state[14:15],
            "left_arm": state[:7],
            "right_gripper": state[15:16],
        }
        return {
            "states": groups,
            "step_index": step,
            "seed": seed,
            "episode": episode,
            "instruction": "pick the red block",
            "images": {
                name: {"shape": [4, 5, 3], "data": bytes([color] * 60)}
                for name, color in (
                    ("head", 25),
                    ("left_hand", 75),
                    ("right_hand", 125),
                )
            },
        }

    def result():
        return {
            "terminated": mode == "terminated" and step >= 2,
            "truncated": step >= config["max_steps"],
            "success": False,
            "official_result": {"code": 0, "scores": {"E2E": 0}},
            "seed": seed,
            "control_steps": step,
        }

    while True:
        request = receive_packet(channel, time.monotonic() + 60)
        method = request["method"]
        if method == "close":
            if mode == "close_timeout":
                time.sleep(60)
            send_packet(
                channel,
                {"id": request["id"], "ok": True, "result": {"closing": True}},
                time.monotonic() + 10,
            )
            raise SystemExit(13 if mode == "close_exit" else 0)
        if method == "reset":
            step, seed = 0, request["payload"]["seed"]
            episode += 1
            response = {
                "observation": observation(),
                "action_limits": [[-3.0, 3.0]] * 16,
                "result": result(),
            }
        elif method == "step":
            with (args.run_directory / "received.jsonl").open("a") as stream:
                stream.write(json.dumps(request) + "\n")
            if mode == "request_exit":
                os._exit(12)
            if mode == "request_timeout":
                time.sleep(60)
            if mode == "backend_error":
                send_packet(
                    channel,
                    {
                        "id": request["id"],
                        "ok": False,
                        "error": "injected native failure",
                    },
                    time.monotonic() + 10,
                )
                continue
            records = []
            for state in request["payload"]["actions"]:
                step += 1
                outcome = result()
                records.append(
                    {
                        "states": observation()["states"],
                        "step_index": step,
                        "reward": 0.0,
                        "terminated": outcome["terminated"],
                        "truncated": outcome["truncated"],
                        "action": state,
                    }
                )
                if outcome["terminated"] or outcome["truncated"]:
                    break
            response = {
                "observation": observation(),
                "steps": records,
                "result": result(),
            }
            if mode == "invalid_final_step":
                response["observation"]["step_index"] += 1
                response["result"]["control_steps"] += 1
        else:
            response = result()
        if mode == "invalid_image" and "observation" in response:
            response["observation"]["images"]["head"]["shape"] = [8, 9, 3]
        send_packet(
            channel,
            {
                "id": request["id"] + (1 if mode == "sequence" else 0),
                "ok": True,
                "result": response,
            },
            time.monotonic() + 10,
        )
