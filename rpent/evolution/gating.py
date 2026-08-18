# Copyright (c) 2026 RPent Contributors
"""Paired candidate gates, including the preregistered two-stage test."""

from __future__ import annotations

import math

from rpent.evolution.jsonio import canonical_sha256
from rpent.evolution.models import EpisodeRecord, GateDecision, GateKind


def one_sided_exact_mcnemar(candidate_wins: int, parent_wins: int) -> float:
    """P[X >= candidate_wins] for X~Binomial(discordant, 0.5)."""

    if candidate_wins < 0 or parent_wins < 0:
        raise ValueError("paired win counts must be non-negative")
    discordant = candidate_wins + parent_wins
    if discordant == 0:
        return 1.0
    numerator = sum(
        math.comb(discordant, value) for value in range(candidate_wins, discordant + 1)
    )
    return numerator / (2**discordant)


def _valid_by_seed(records: list[EpisodeRecord]) -> dict[int, EpisodeRecord]:
    result: dict[int, EpisodeRecord] = {}
    for record in records:
        if record.status != "valid":
            continue
        if record.seed in result:
            raise ValueError(f"duplicate valid paired seed: {record.seed}")
        result[record.seed] = record
    return result


def _assert_bundle_binding(
    records: dict[int, EpisodeRecord],
    *,
    expected_sha256: str | None,
    arm: str,
) -> None:
    mismatched = sorted(
        seed
        for seed, record in records.items()
        if record.bundle_sha256 != expected_sha256
    )
    if mismatched:
        raise ValueError(
            f"{arm} gate records are not bound to the declared bundle: {mismatched}"
        )


def _action_artifact_sha(record: EpisodeRecord) -> str | None:
    trajectory = record.artifact_index.get("trajectory_index")
    if not isinstance(trajectory, dict):
        return None
    hashes = trajectory.get("artifact_sha256")
    if not isinstance(hashes, dict):
        return None
    value = hashes.get("actions")
    return str(value) if isinstance(value, str) else None


def _candidate_intervened(record: EpisodeRecord) -> bool:
    """Return whether the learned Critic/Role1 path actually ran.

    ``candidate_intervention`` is the forward-compatible attestation.  The
    Role1 artifact prefix keeps already-completed candidate episodes
    auditable without rewriting their immutable EpisodeRecords.
    """

    declared = record.artifact_index.get("candidate_intervention")
    if declared is not None:
        return bool(declared)
    return any(str(key).startswith("role1:") for key in record.artifact_index)


def _same_physical_reset(
    candidate_identity: dict[str, object], parent_identity: dict[str, object]
) -> tuple[bool, bool]:
    """Compare the recorded reset state and separately audit renderer drift.

    Exact camera byte hashes are not a stable reset identity across EGL
    contexts.  The environment seed and policy RNG are checked by the caller;
    when both records expose a state digest, the preregistered environment
    seed plus that digest form the stable reset binding. Camera digests remain
    immutable evidence and their mismatch count is reported in the rationale.

    Historical records without ``state_sha256`` retain the old full-object
    comparison rather than silently weakening their contract.
    """

    candidate_state = candidate_identity.get("state_sha256")
    parent_state = parent_identity.get("state_sha256")
    if candidate_state is None and parent_state is None:
        return (
            canonical_sha256(candidate_identity) == canonical_sha256(parent_identity),
            False,
        )
    if not isinstance(candidate_state, str) or not candidate_state:
        raise ValueError("candidate reset identity has no physical state digest")
    if not isinstance(parent_state, str) or not parent_state:
        raise ValueError("parent reset identity has no physical state digest")
    camera_mismatch = canonical_sha256(
        candidate_identity.get("camera_sha256", {})
    ) != canonical_sha256(parent_identity.get("camera_sha256", {}))
    return candidate_state == parent_state, camera_mismatch


def _assert_paired_evidence(
    candidate: EpisodeRecord, parent: EpisodeRecord, *, require_divergence: bool
) -> tuple[bool, bool]:
    if candidate.policy_rng != parent.policy_rng:
        raise ValueError("paired gate requires identical policy RNG per seed")
    candidate_identity = candidate.artifact_index.get("initial_observation_identity")
    parent_identity = parent.artifact_index.get("initial_observation_identity")
    if not isinstance(candidate_identity, dict) or not candidate_identity:
        raise ValueError("candidate is missing initial observation identity")
    if not isinstance(parent_identity, dict) or not parent_identity:
        raise ValueError("parent is missing initial observation identity")
    same_reset, camera_mismatch = _same_physical_reset(
        candidate_identity, parent_identity
    )
    if not same_reset:
        raise ValueError("paired gate initial observation identity differs")
    if require_divergence:
        candidate_actions = _action_artifact_sha(candidate)
        parent_actions = _action_artifact_sha(parent)
        if not candidate_actions or not parent_actions:
            raise ValueError("same-seed gate requires action trajectory digests")
        return candidate_actions != parent_actions, camera_mismatch
    return False, camera_mismatch


def evaluate_paired_gate(
    *,
    kind: GateKind,
    candidate_sha256: str,
    parent_sha256: str | None,
    candidate_records: list[EpisodeRecord],
    parent_records: list[EpisodeRecord],
    expected_seeds: tuple[int, ...],
    alpha: float | None = None,
    same_seed_pass_rate: float = 1.0,
    heldout_min_gain: int = 1,
    heldout_min_success_rate: float = 0.0,
    heldout_require_significance: bool = True,
) -> GateDecision:
    candidate = _valid_by_seed(candidate_records)
    parent = _valid_by_seed(parent_records)
    _assert_bundle_binding(
        candidate,
        expected_sha256=candidate_sha256,
        arm="candidate",
    )
    _assert_bundle_binding(parent, expected_sha256=parent_sha256, arm="parent")
    expected = set(expected_seeds)
    if set(candidate) != expected or set(parent) != expected:
        missing_candidate = sorted(expected - set(candidate))
        missing_parent = sorted(expected - set(parent))
        raise ValueError(
            "paired gate requires exactly one valid result per seed; "
            f"candidate_missing={missing_candidate}, parent_missing={missing_parent}"
        )
    if not 0 < same_seed_pass_rate <= 1:
        raise ValueError("same-seed pass rate must be in (0, 1]")
    divergence_by_seed: dict[int, bool] = {}
    camera_mismatch_by_seed: dict[int, bool] = {}
    intervention_by_seed: dict[int, bool] = {}
    for seed in expected_seeds:
        divergence, camera_mismatch = _assert_paired_evidence(
            candidate[seed],
            parent[seed],
            require_divergence=kind == "same_seed",
        )
        divergence_by_seed[seed] = divergence
        camera_mismatch_by_seed[seed] = camera_mismatch
        intervention_by_seed[seed] = _candidate_intervened(candidate[seed])
    mechanism_diverged = any(
        divergence_by_seed[seed] and intervention_by_seed[seed]
        for seed in expected_seeds
    )
    candidate_successes = sum(bool(candidate[seed].success) for seed in expected_seeds)
    parent_successes = sum(bool(parent[seed].success) for seed in expected_seeds)
    candidate_wins = sum(
        bool(candidate[seed].success) and not bool(parent[seed].success)
        for seed in expected_seeds
    )
    parent_wins = sum(
        bool(parent[seed].success) and not bool(candidate[seed].success)
        for seed in expected_seeds
    )
    candidate_safety = sum(
        len(candidate[seed].safety_events) for seed in expected_seeds
    )
    parent_safety = sum(len(parent[seed].safety_events) for seed in expected_seeds)
    no_safety_regression = candidate_safety <= parent_safety

    if kind == "same_seed":
        conclusive = True
        if same_seed_pass_rate == 1.0:
            # Preserve the byte-for-byte semantics of historical v1 plans.
            required_successes = len(expected_seeds)
            passed = (
                mechanism_diverged
                and candidate_successes == required_successes
                and parent_successes == 0
                and no_safety_regression
            )
        else:
            required_successes = math.ceil(
                len(expected_seeds) * same_seed_pass_rate
            )
            rescued_seeds = tuple(
                seed
                for seed in expected_seeds
                if candidate[seed].success and not parent[seed].success
            )
            rescued_diverged = bool(rescued_seeds) and all(
                divergence_by_seed[seed] and intervention_by_seed[seed]
                for seed in rescued_seeds
            )
            passed = (
                candidate_successes >= required_successes
                and candidate_successes > parent_successes
                and rescued_diverged
                and no_safety_regression
            )
        p_value = None
        rationale = (
            "candidate never changed the failed parent action trajectory"
            if not mechanism_diverged
            else (
                f"candidate met the frozen same-seed threshold: "
                f"{candidate_successes}/{len(expected_seeds)} >= "
                f"{required_successes}/{len(expected_seeds)}"
            )
            if passed
            else (
                f"candidate did not meet the frozen same-seed improvement threshold "
                f"({candidate_successes}/{len(expected_seeds)}, required "
                f"{required_successes}), rescued trajectories did not diverge, "
                "or safety regressed"
            )
        )
    elif kind == "regression":
        conclusive = True
        passed = candidate_successes >= parent_successes and no_safety_regression
        p_value = None
        rationale = (
            "candidate did not reduce success or safety"
            if passed
            else ("candidate regressed success or safety")
        )
    else:
        if alpha is None or not 0 < alpha < 1:
            raise ValueError("heldout gate requires alpha in (0, 1)")
        if heldout_min_gain < 0:
            raise ValueError("heldout_min_gain must be non-negative")
        if not 0 <= heldout_min_success_rate <= 1:
            raise ValueError("heldout_min_success_rate must be in [0, 1]")
        if not isinstance(heldout_require_significance, bool):
            raise ValueError("heldout_require_significance must be boolean")
        p_value = one_sided_exact_mcnemar(candidate_wins, parent_wins)
        conclusive = p_value < alpha if heldout_require_significance else True
        success_rate = candidate_successes / len(expected_seeds)
        gain = candidate_successes - parent_successes
        passed = (
            conclusive
            and gain >= heldout_min_gain
            and success_rate >= heldout_min_success_rate
            and no_safety_regression
        )
        rationale = (
            f"paired one-sided exact McNemar p={p_value:.6g}; "
            f"alpha={alpha}; require_significance={heldout_require_significance}; "
            f"gain={gain}; min_gain={heldout_min_gain}; "
            f"success_rate={success_rate:.6g}; "
            f"min_success_rate={heldout_min_success_rate:.6g}; "
            f"no_safety_regression={no_safety_regression}"
        )

    camera_mismatches = sum(camera_mismatch_by_seed.values())
    if camera_mismatches:
        rationale += (
            f"; reset state binding matched for all pairs while exact camera "
            f"digests differed for {camera_mismatches}/{len(expected_seeds)} pairs"
        )

    decision_payload = {
        "kind": kind,
        "candidate": candidate_sha256,
        "parent": parent_sha256,
        "seeds_digest": canonical_sha256(list(expected_seeds)),
        "candidate_successes": candidate_successes,
        "parent_successes": parent_successes,
        "mechanism_diverged": mechanism_diverged,
    }
    if kind.startswith("heldout") and (
        alpha != 0.025
        or heldout_min_gain != 1
        or heldout_min_success_rate != 0.0
        or not heldout_require_significance
    ):
        decision_payload["heldout_policy"] = {
            "alpha": alpha,
            "min_gain": heldout_min_gain,
            "min_success_rate": heldout_min_success_rate,
            "require_significance": heldout_require_significance,
        }
    return GateDecision(
        decision_id=f"gate-{canonical_sha256(decision_payload)[:20]}",
        candidate_sha256=candidate_sha256,
        parent_sha256=parent_sha256,
        kind=kind,
        passed=passed,
        conclusive=conclusive,
        candidate_successes=candidate_successes,
        parent_successes=parent_successes,
        paired_count=len(expected_seeds),
        candidate_wins=candidate_wins,
        parent_wins=parent_wins,
        p_value=p_value,
        alpha=alpha,
        candidate_safety_events=candidate_safety,
        parent_safety_events=parent_safety,
        rationale=rationale,
    )


def evaluate_two_stage_heldout(
    *,
    candidate_sha256: str,
    parent_sha256: str | None,
    candidate_records: list[EpisodeRecord],
    parent_records: list[EpisodeRecord],
    preregistered_seeds: tuple[int, ...],
    stage: int,
    alpha: float = 0.025,
    min_gain: int = 1,
    min_success_rate: float = 0.0,
    require_significance: bool = True,
) -> GateDecision:
    if len(preregistered_seeds) != 50:
        raise ValueError("two-stage heldout gate requires 50 preregistered seeds")
    if stage == 1:
        seeds = preregistered_seeds[:10]
        kind: GateKind = "heldout_10"
    elif stage == 2:
        seeds = preregistered_seeds
        kind = "heldout_50"
    else:
        raise ValueError("heldout stage must be 1 or 2")
    return evaluate_paired_gate(
        kind=kind,
        candidate_sha256=candidate_sha256,
        parent_sha256=parent_sha256,
        candidate_records=candidate_records,
        parent_records=parent_records,
        expected_seeds=seeds,
        alpha=alpha,
        heldout_min_gain=min_gain,
        heldout_min_success_rate=min_success_rate,
        heldout_require_significance=require_significance,
    )


def evaluate_fixed_heldout_20(
    *,
    candidate_sha256: str,
    parent_sha256: str | None,
    candidate_records: list[EpisodeRecord],
    parent_records: list[EpisodeRecord],
    preregistered_seeds: tuple[int, ...],
    alpha: float = 0.025,
    min_gain: int = 1,
    min_success_rate: float = 0.0,
    require_significance: bool = True,
) -> GateDecision:
    """Evaluate one immutable, paired 20-seed unseen block.

    This is intentionally separate from the legacy 10→50 sequential gate:
    a 20-seed report must never be mislabeled as either stage of that contract.
    """

    if len(preregistered_seeds) != 20:
        raise ValueError("fixed heldout_20 gate requires exactly 20 preregistered seeds")
    return evaluate_paired_gate(
        kind="heldout_20",
        candidate_sha256=candidate_sha256,
        parent_sha256=parent_sha256,
        candidate_records=candidate_records,
        parent_records=parent_records,
        expected_seeds=preregistered_seeds,
        alpha=alpha,
        heldout_min_gain=min_gain,
        heldout_min_success_rate=min_success_rate,
        heldout_require_significance=require_significance,
    )
