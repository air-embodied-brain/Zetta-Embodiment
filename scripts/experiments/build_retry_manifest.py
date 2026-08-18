"""Build an auditable retry manifest from selected pairs in a frozen manifest."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_retry(
    source: dict[str, Any],
    *,
    source_path: Path,
    pair_ids: list[str],
    retry_tag: str,
    output_root: Path,
    reason: str,
    purpose: str = "paired retry",
    commit_overrides: dict[str, str] | None = None,
    tool_sha_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not retry_tag or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for char in retry_tag):
        raise ValueError("retry_tag must contain only lowercase letters, digits, '-' or '_'")
    requested = set(pair_ids)
    selected = [episode for episode in source["episodes"] if episode.get("pair_id") in requested]
    found = {episode.get("pair_id") for episode in selected}
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"pair IDs are absent from the source manifest: {missing}")
    counts = Counter(episode.get("pair_id") for episode in selected)
    malformed = {pair_id: count for pair_id, count in counts.items() if count != 2}
    if malformed:
        raise ValueError(f"retry requires exactly two variants per pair: {malformed}")

    retry = copy.deepcopy(source)
    retry["protocol_id"] = f'{source["protocol_id"]}-{retry_tag}'
    retry["retry"] = {
        "tag": retry_tag,
        "reason": reason,
        "source_manifest": str(source_path.resolve()),
        "source_manifest_sha256": _sha256(source_path),
        "original_pair_ids": pair_ids,
        "commit_overrides": commit_overrides or {},
        "tool_sha_overrides": tool_sha_overrides or {},
    }
    retry["selection"] = {
        "pairs": len(pair_ids),
        "episodes": len(selected),
        "purpose": purpose,
    }
    retry_episodes: list[dict[str, Any]] = []
    for episode in selected:
        original_pair_id = str(episode["pair_id"])
        rewritten = copy.deepcopy(episode)
        rewritten["original_episode_id"] = episode["id"]
        rewritten["original_pair_id"] = original_pair_id
        rewritten["id"] = f'{retry_tag}-{episode["id"]}'
        rewritten["pair_id"] = f"{retry_tag}-{original_pair_id}"
        rewritten["phase"] = retry_tag
        variant = str(episode["variant"])
        if commit_overrides and variant in commit_overrides:
            rewritten["original_commit"] = episode["commit"]
            rewritten["commit"] = commit_overrides[variant]
        if tool_sha_overrides and variant in tool_sha_overrides:
            rewritten["original_tool_manifest_sha256"] = episode[
                "tool_manifest_sha256"
            ]
            rewritten["tool_manifest_sha256"] = tool_sha_overrides[variant]
        rewritten["output_dir"] = str(
            (output_root / retry_tag / original_pair_id / str(episode["variant"])).resolve()
        )
        retry_episodes.append(rewritten)
    retry["episodes"] = retry_episodes
    return retry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--retry-tag", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--purpose", default="paired retry")
    parser.add_argument("--pair-id", action="append", required=True, dest="pair_ids")
    parser.add_argument(
        "--variant-commit",
        action="append",
        default=[],
        metavar="VARIANT=SHA",
        help="override the expected commit for a retry variant",
    )
    parser.add_argument(
        "--variant-tool-sha",
        action="append",
        default=[],
        metavar="VARIANT=SHA",
        help="override the robots/libero tree SHA for a retry variant",
    )
    args = parser.parse_args()

    commit_overrides: dict[str, str] = {}
    for value in args.variant_commit:
        variant, separator, commit = value.partition("=")
        if not separator or not variant or len(commit) != 40:
            raise ValueError(f"invalid --variant-commit value: {value!r}")
        commit_overrides[variant] = commit
    tool_sha_overrides: dict[str, str] = {}
    for value in args.variant_tool_sha:
        variant, separator, tool_sha = value.partition("=")
        if not separator or not variant or len(tool_sha) != 40:
            raise ValueError(f"invalid --variant-tool-sha value: {value!r}")
        tool_sha_overrides[variant] = tool_sha

    source = json.loads(args.source.read_text(encoding="utf-8"))
    retry = build_retry(
        source,
        source_path=args.source,
        pair_ids=args.pair_ids,
        retry_tag=args.retry_tag,
        output_root=args.output_root,
        reason=args.reason,
        purpose=args.purpose,
        commit_overrides=commit_overrides,
        tool_sha_overrides=tool_sha_overrides,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(retry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(retry["selection"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
