# Copyright (c) 2026 Zetta Contributors
"""Check headless Isaac Sim rendering and physics before loading task assets."""

from __future__ import annotations

import argparse
import faulthandler
import gc
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--active-gpu", type=int, default=0)
    parser.add_argument("--physics-gpu", type=int, default=0)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_file = output / "result.json"
    if not args.child:
        result_file.write_text(
            json.dumps({"status": "initializing", "shutdown_complete": False}) + "\n"
        )
        try:
            process = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    *sys.argv[1:],
                    "--child",
                ],
                check=False,
                timeout=840,
            )
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            returncode = 124
        result = json.loads(result_file.read_text())
        passed = (
            returncode == 0
            and result.get("status") == "render_physics_passed"
            and result.get("scene_cleanup_complete") is True
            and result.get("close_requested") is True
        )
        result.update(
            status="passed" if passed else "failed",
            shutdown_complete=passed,
            child_returncode=returncode,
            elapsed_seconds=time.monotonic() - started,
        )
        result_file.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if not passed:
            raise SystemExit(
                "Isaac preflight failed; see result.json and the process log"
            )
        return
    faulthandler.dump_traceback_later(90, repeat=True)

    def milestone(stage: str) -> None:
        print(
            json.dumps({"stage": stage, "elapsed_seconds": time.monotonic() - started}),
            flush=True,
        )

    token_paths = {
        "data": output / "kit" / "data",
        "cache": output / "kit" / "cache",
        "logs": output / "kit" / "logs",
        "documents": output / "kit" / "documents",
    }
    for path in token_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    tokens = dict(token_paths)
    tokens.update({f"omni_{name}": path for name, path in token_paths.items()})
    tokens["shared_documents"] = token_paths["documents"]
    extra_args = [f"--/app/tokens/{name}={path}" for name, path in tokens.items()]
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        extra_args.append("--allow-root")

    from isaacsim import SimulationApp

    milestone("starting_simulation_app")
    app = SimulationApp(
        {
            "headless": True,
            "fast_shutdown": True,
            "active_gpu": args.active_gpu,
            "physics_gpu": args.physics_gpu,
            "multi_gpu": False,
            "renderer": "RaytracedLighting",
            "width": 256,
            "height": 256,
            "limit_cpu_threads": 8,
            "extra_args": extra_args,
        }
    )
    milestone("simulation_app_ready")
    camera = world = cube = light = rgba = None
    result = None
    try:
        import numpy as np
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid
        from isaacsim.core.utils.viewports import set_camera_view
        from isaacsim.sensors.camera import Camera
        from PIL import Image
        from pxr import UsdLux

        world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)
        milestone("world_created")
        world.scene.add_default_ground_plane()
        milestone("ground_created")
        cube = world.scene.add(
            DynamicCuboid(
                prim_path="/World/ProbeCube",
                name="probe_cube",
                position=np.array([0.0, 0.0, 0.8]),
                scale=np.array([0.2, 0.2, 0.2]),
                color=np.array([0.8, 0.1, 0.05]),
                mass=0.1,
            )
        )
        light = UsdLux.DomeLight.Define(world.stage, "/World/ProbeLight")
        light.CreateIntensityAttr(1000.0)
        camera = Camera(prim_path="/World/ProbeCamera", resolution=(256, 256))
        set_camera_view(
            eye=np.array([1.2, 1.2, 0.9]),
            target=np.array([0.0, 0.0, 0.2]),
            camera_prim_path="/World/ProbeCamera",
        )
        world.reset()
        camera.initialize()
        milestone("camera_initialized")
        initial_z = float(cube.get_world_pose()[0][2])
        for _ in range(120):
            world.step(render=True)
        rgba = np.asarray(camera.get_rgba())
        if rgba.shape != (256, 256, 4):
            raise RuntimeError(f"camera returned invalid shape: {rgba.shape}")
        rgb = np.ascontiguousarray(rgba[:, :, :3], dtype=np.uint8)
        final_z = float(cube.get_world_pose()[0][2])
        result = {
            "status": "render_physics_pending",
            "shutdown_complete": False,
            "isaacsim_version": importlib.metadata.version("isaacsim"),
            "active_gpu": args.active_gpu,
            "physics_gpu": args.physics_gpu,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "image_shape": list(rgb.shape),
            "image_std": float(rgb.std()),
            "image_unique_colors": int(np.unique(rgb.reshape(-1, 3), axis=0).shape[0]),
            "initial_cube_z": initial_z,
            "final_cube_z": final_z,
            "physics_steps": int(world.current_time_step_index),
        }
        Image.fromarray(rgb).save(output / "frame.png")
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        if result["image_std"] < 2 or result["image_unique_colors"] < 10:
            raise RuntimeError(f"rendered image is blank: {result}")
        if not (initial_z - final_z > 0.4 and 0.05 < final_z < 0.2):
            raise RuntimeError(f"cube did not fall and settle on the floor: {result}")
        result["status"] = "render_physics_passed"
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        milestone("render_physics_passed")
    except BaseException:
        if result is None:
            result = {"shutdown_complete": False}
        result.update(status="failed", error=traceback.format_exc())
        result_file.write_text(json.dumps(result, indent=2) + "\n")
        traceback.print_exc()
        raise
    finally:
        milestone("releasing_scene")
        if world is not None:
            world.stop()
        if camera is not None:
            destroy = getattr(camera, "destroy", None)
            if destroy is not None:
                destroy()
            destroy = None
        # Sensor destructors reference Kit plugins; collect before unloading them.
        camera = cube = light = rgba = None
        if world is not None:
            world.clear_instance()
            world = None
        gc.collect()
        if result is not None:
            result.update(scene_cleanup_complete=True, close_requested=True)
            result_file.write_text(json.dumps(result, indent=2) + "\n")
        milestone("closing_simulation_app")
        # Kit's default close exits the process; the parent confirms its exit.
        app.close()
        faulthandler.cancel_dump_traceback_later()
        milestone("simulation_app_closed")


if __name__ == "__main__":
    main()
