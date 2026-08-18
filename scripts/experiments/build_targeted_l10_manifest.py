"""Build the frozen targeted L10 baseline-vs-integrated experiment matrix."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from liberopro.liberopro.benchmark import get_benchmark

from scripts.experiments.run_paired_episode import _sha256_file, _sha256_tree


# Targeted cells from RPENT_ORIGINAL_L10_STRICT_SEED_MATRIX_20260801.md.
# This intentionally over-samples historically hard cells and includes two
# high-success anchors; it is an engineering validation, not a replacement for
# the paper's complete 200-episode matrix.
CELLS: tuple[tuple[str, int, tuple[int, ...]], ...] = (
    ("libero_10_task", 3, (1, 2, 3, 4)),
    ("libero_10_task", 5, (2, 5, 6)),
    ("libero_10_task", 9, (1, 2, 3, 4)),
    ("libero_10_swap", 3, (1, 2, 3)),
    ("libero_10_swap", 5, (1, 2, 3)),
    ("libero_10_swap", 8, (1, 2, 3, 4)),
    ("libero_10_swap", 9, (1, 2, 3, 4)),
)

HISTORICAL: dict[tuple[str, int], tuple[int, ...]] = {
    ("libero_10_task", 3): (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ("libero_10_task", 5): (1, 1, 1, 1, 0, 1, 1, 1, 1, 1),
    ("libero_10_task", 9): (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ("libero_10_swap", 3): (0, 0, 0, 0, 0, 1, 0, 1, 0, 0),
    ("libero_10_swap", 5): (1, 0, 1, 1, 1, 1, 1, 0, 1, 1),
    ("libero_10_swap", 8): (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ("libero_10_swap", 9): (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
}

GPU_PAIRS = ((0, 1), (3, 2), (4, 5), (7, 6))


def _task_files(libero_root: Path, suite: str, task_id: int) -> list[dict[str, str]]:
    benchmark = get_benchmark(suite)()
    task = benchmark.get_task(task_id)
    paths = (
        libero_root / "bddl_files" / task.problem_folder / task.bddl_file,
        libero_root / "init_files" / task.problem_folder / task.init_states_file,
    )
    return [{"path": str(path.resolve()), "sha256": _sha256_file(path)} for path in paths]


def build(args: argparse.Namespace) -> dict[str, Any]:
    baseline_repo = args.baseline_repo.resolve()
    integrated_repo = args.integrated_repo.resolve()
    baseline_resources = baseline_repo / "resources" / "libero"
    integrated_resources = integrated_repo / "resources" / "libero"
    baseline_tree = _sha256_tree(baseline_resources)
    integrated_tree = _sha256_tree(integrated_resources)
    if baseline_tree != integrated_tree:
        raise RuntimeError(
            f"baseline/integrated resource trees differ: {baseline_tree} vs {integrated_tree}"
        )

    import liberopro.liberopro as liberopro_package

    libero_root = Path(liberopro_package.__file__).resolve().parent
    assets_root = (libero_root / "assets").resolve()
    assets_tree = _sha256_tree(assets_root)
    episodes: list[dict[str, Any]] = []
    pair_index = 0
    for suite, task_id, seeds in CELLS:
        task_files = _task_files(libero_root, suite, task_id)
        for seed in seeds:
            gpu_baseline, gpu_integrated = GPU_PAIRS[pair_index % len(GPU_PAIRS)]
            if (pair_index // len(GPU_PAIRS)) % 2:
                gpu_baseline, gpu_integrated = gpu_integrated, gpu_baseline
            short_suite = suite.removeprefix("libero_")
            pair_id = f"formal-{short_suite}-t{task_id}-s{seed}"
            historical = bool(HISTORICAL[(suite, task_id)][seed - 1])
            for variant, repo, commit, tool_sha, gpu in (
                (
                    "baseline",
                    baseline_repo,
                    args.baseline_commit,
                    args.baseline_tool_sha,
                    gpu_baseline,
                ),
                (
                    "integrated",
                    integrated_repo,
                    args.integrated_commit,
                    args.integrated_tool_sha,
                    gpu_integrated,
                ),
            ):
                output_dir = (
                    args.output_root
                    / "formal"
                    / short_suite
                    / f"t{task_id}-s{seed}"
                    / variant
                )
                resource_root = repo / "resources" / "libero"
                episodes.append(
                    {
                        "id": f"{pair_id}-{variant}",
                        "pair_id": pair_id,
                        "phase": "formal",
                        "variant": variant,
                        "suite": suite,
                        "task": task_id,
                        "seed": seed,
                        "historical_original_success": historical,
                        "repo": str(repo),
                        "commit": commit,
                        "output_dir": str(output_dir),
                        "cuda_device": gpu,
                        "tool_manifest_sha256": tool_sha,
                        "extra_args": ["--libero-type", "pro"],
                        "resource_snapshot": {
                            "files": task_files,
                            "trees": [
                                {
                                    "path": str(resource_root),
                                    "sha256": baseline_tree[0],
                                    "file_count": baseline_tree[1],
                                },
                                {
                                    "path": str(assets_root),
                                    "sha256": assets_tree[0],
                                    "file_count": assets_tree[1],
                                },
                            ],
                        },
                    }
                )
            pair_index += 1

    selected_historical = sum(
        HISTORICAL[(suite, task)][seed - 1]
        for suite, task, seeds in CELLS
        for seed in seeds
    )
    return {
        "schema_version": 1,
        "protocol_id": "rpent-l10-targeted-paired-20260801-v2",
        "ledger_path": str(args.ledger.resolve()),
        "budget_cap": 100,
        # Preserve the virtualenv symlink; resolving it drops venv site-packages.
        "python": str(args.python.absolute()),
        "common": {
            "model": "gpt-5.6-terra",
            "reasoning_effort": "medium",
            "max_turns": 100,
            "max_episode_steps": 10000,
            "planner_timeout_s": 1800,
            "episode_timeout_s": 3600,
            "disable_sam3": True,
            "hf_hub_offline": True,
        },
        "selection": {
            "pairs": pair_index,
            "episodes": len(episodes),
            "historical_selected_successes": selected_historical,
            "historical_selected_rate": selected_historical / pair_index,
            "full_original_matrix_rate": 0.44,
            "purpose": "targeted paired engineering validation",
        },
        "model_provenance": {
            "pi05_checkpoint": str(args.pi05_checkpoint.resolve()),
            "model_safetensors_sha256": args.pi05_model_sha256,
            "assets_tree_sha256": assets_tree[0],
            "assets_file_count": assets_tree[1],
        },
        "episodes": episodes,
    }


def main() -> int:
    base = Path(
        os.environ.get("RPENT_DEPLOY_ROOT", Path(__file__).resolve().parents[2])
    ).resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=base / "artifacts" / "paired-100")
    parser.add_argument("--ledger", type=Path, default=base / "artifacts" / "paired-100" / "episode_ledger.jsonl")
    parser.add_argument("--python", type=Path, default=base / ".venv-fast" / "bin" / "python")
    parser.add_argument("--baseline-repo", type=Path, default=base / "rpent-baseline-run")
    parser.add_argument("--integrated-repo", type=Path, default=base / "rpent-integrated-run")
    parser.add_argument("--baseline-commit", default="a679b0f99ebc8b3c60cbb0025a5bc6ec50a56309")
    parser.add_argument("--integrated-commit", default="b51cf9c5f83eb188ae0eb38613ce1273710c083d")
    parser.add_argument("--baseline-tool-sha", default="865146f876fcb6ed91bf3976bf9cabbfeef1cedd")
    parser.add_argument("--integrated-tool-sha", default="d49a6aae3933ad25431f291b14818ca3aa01cf18")
    parser.add_argument("--pi05-checkpoint", type=Path, default=base / "checkpoints" / "pi05")
    parser.add_argument("--pi05-model-sha256", default="4d9089c941793f170b625c2ed0ac7a3aa09b6f103e52dbbc82e67301529d6683")
    args = parser.parse_args()
    manifest = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["selection"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
