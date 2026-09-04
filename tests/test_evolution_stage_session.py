# Copyright (c) 2026 Zetta Contributors
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from zetta.evolution.jsonio import atomic_write_json, canonical_sha256, read_json
from zetta.evolution.models import CausalDiagnosis, FailureCluster
from zetta.evolution.stages import (
    PROPOSAL_SYSTEM_PROMPT,
    CodexStageAgent,
    _candidate_from_payload,
    _diagnostic_telemetry_contract,
    _extract_json_object,
    _normalize_task_contract,
    _reject_collision_control,
    _validate_recovery_chunk_policy,
    _validate_recovery_tool_parameters,
)
from zetta.planner.base import PlannerResult


def _cluster() -> FailureCluster:
    return FailureCluster(
        cluster_id="cluster-stall",
        hard_key=("engage", "robocasa.vla.groot", "stall"),
        member_segment_ids=("artifact-" + "1" * 64,),
        episode_ids=("artifact-" + "2" * 64,),
        representative_segment_ids=("artifact-" + "1" * 64,),
        medoid_segment_id="artifact-" + "1" * 64,
        summary="contact progress stalls",
        prevalence=0.7,
        mean_severity=0.8,
    )


def _artifact_index() -> dict[str, Any]:
    return {
        "artifacts": [
            {
                "content_id": "artifact-" + "1" * 64,
                "type": "failure_segment",
                "summary": "failure-segment evidence",
                "hash": "hmac-sha256:" + "2" * 64,
            }
        ]
    }


def test_diagnosis_telemetry_contract_and_access_are_audited(tmp_path: Path) -> None:
    failure_trace = "artifact-" + "3" * 64
    success_trace = "artifact-" + "4" * 64
    index = {
        **_artifact_index(),
        "diagnostic_telemetry": [
            {
                "episode": _cluster().episode_ids[0],
                "outcome": "failure",
                "content_id": failure_trace,
            },
            {
                "episode": "artifact-" + "5" * 64,
                "outcome": "success",
                "content_id": success_trace,
            },
        ],
    }
    contract = _diagnostic_telemetry_contract(
        cluster=_cluster(), artifact_index=index
    )
    assert contract["required_content_ids"] == [failure_trace, success_trace]

    access_log = tmp_path / "evidence-access.jsonl"
    access_log.write_text(
        json.dumps({"content_id": failure_trace, "kind": "structured"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="did not read required"):
        CodexStageAgent._validate_structured_access(
            access_log,
            required_content_ids={failure_trace, success_trace},
        )
    with access_log.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps({"content_id": success_trace, "kind": "structured"}) + "\n"
        )
    CodexStageAgent._validate_structured_access(
        access_log,
        required_content_ids={failure_trace, success_trace},
    )


def _diagnosis_payload() -> dict[str, Any]:
    return {
        "diagnosis_id": "diagnosis-stall",
        "cluster_id": "cluster-stall",
        "outcome": "the rack remains closed at the horizon",
        "immediate_trigger": "repeated contact without task progress",
        "root_cause": "the contact transition does not produce progress",
        "contributing_causes": [],
        "competing_hypotheses": [
            "the contact transition does not produce progress",
            "the critic interrupts a valid transition too early",
        ],
        "owner_layer": "critic",
        "affected_component": "rack engagement critic",
        "earliest_divergence": "first repeated no-progress transition",
        "supporting_evidence_ids": ["artifact-" + "1" * 64],
        "counterevidence_ids": [],
        "falsifier": "progress resumes without intervention",
        "distinguishing_check": "paired recovery changes progress",
        "required_validation": "paired live same-state intervention",
        "confidence": 0.8,
    }


def _candidate_payload() -> dict[str, Any]:
    evidence = "artifact-" + "1" * 64
    return {
        "candidate_id": "candidate-stall",
        "mechanism_change": "add a bounded progress-stall proposal",
        "validation_plan": "pair parent and candidate on every cluster seed",
        "critic_rules": [
            {
                "rule_id": "progress-stall",
                "title": "detect retained-contact stall",
                "feature": "privileged.progress",
                "operator": "stagnant",
                "threshold": 0.01,
                "dwell_steps": 3,
                "cooldown_steps": 2,
                "proposal": "request bounded re-engagement",
                "evidence_ids": [evidence],
                "safety_only": False,
            }
        ],
        "recovery_rules": [
            {
                "recovery_id": "reengage",
                "title": "bounded re-engagement",
                "trigger_rule_ids": ["progress-stall"],
                "precondition": "the proposal is current",
                "steps": [
                    {
                        "tool": "robocasa.vla.groot",
                        "parameters": {},
                        "stop_when": "a fresh state is available",
                    }
                ],
                "safety_constraints": ["Actor remains the only writer"],
                "stop_condition": "one bounded proposal is consumed",
                "fallback": "stop safely",
                "evidence_ids": [evidence],
            }
        ],
        "tool_plugin": None,
    }


def test_collision_telemetry_cannot_control_candidate() -> None:
    diagnosis = CausalDiagnosis(**_diagnosis_payload())
    payload = _candidate_payload()
    payload["critic_rules"][0]["activation_conditions"] = [
        {
            "feature": "privileged.dishwasher.collision.detected",
            "operator": "eq",
            "threshold": True,
        }
    ]
    candidate = _candidate_from_payload(
        payload,
        generation=0,
        parent_sha256=None,
        diagnosis=diagnosis,
    )
    with pytest.raises(ValueError, match="diagnostic-only"):
        _reject_collision_control(candidate)


def test_recovery_parameters_must_match_frozen_tool_schema() -> None:
    diagnosis = CausalDiagnosis(**_diagnosis_payload())
    payload = _candidate_payload()
    payload["recovery_rules"][0]["steps"][0]["parameters"] = {
        "unsupported_runtime_kwarg": True
    }
    candidate = _candidate_from_payload(
        payload,
        generation=0,
        parent_sha256=None,
        diagnosis=diagnosis,
    )
    catalog = {
        "tools": [
            {
                "name": "robocasa.vla.groot",
                "input_schema": {
                    "type": "object",
                    "properties": {"action_horizon": {"type": "integer"}},
                },
            }
        ]
    }
    with pytest.raises(ValueError, match="outside its frozen schema"):
        _validate_recovery_tool_parameters(candidate, catalog)


def test_collision_aware_planner_name_is_not_mistaken_for_detector_gate() -> None:
    diagnosis = CausalDiagnosis(**_diagnosis_payload())
    payload = _candidate_payload()
    payload["recovery_rules"][0]["steps"][0]["tool"] = (
        "robocasa.motion.collision_aware_local"
    )
    candidate = _candidate_from_payload(
        payload,
        generation=0,
        parent_sha256=None,
        diagnosis=diagnosis,
    )
    _reject_collision_control(candidate)


def test_safety_constraint_may_explicitly_forbid_collision_detector_control() -> None:
    diagnosis = CausalDiagnosis(**_diagnosis_payload())
    payload = _candidate_payload()
    payload["recovery_rules"][0]["safety_constraints"] = [
        "Do not use collision-detector output for activation or stopping."
    ]
    candidate = _candidate_from_payload(
        payload,
        generation=0,
        parent_sha256=None,
        diagnosis=diagnosis,
    )
    _reject_collision_control(candidate)


def test_vla_recovery_one_action_chunk_requires_latency_justification() -> None:
    diagnosis = CausalDiagnosis(**_diagnosis_payload())
    payload = _candidate_payload()
    payload["recovery_rules"][0]["steps"][0]["tool"] = "libero.vla_execute"
    payload["recovery_rules"][0]["steps"][0]["parameters"] = {
        "actions_per_chunk": 1
    }
    candidate = _candidate_from_payload(
        payload,
        generation=0,
        parent_sha256=None,
        diagnosis=diagnosis,
    )
    with pytest.raises(ValueError, match="per-action boundary"):
        _validate_recovery_chunk_policy(candidate)
    candidate = _candidate_from_payload(
        {**payload, "validation_plan": "paired per-action boundary recovery"},
        generation=0,
        parent_sha256=None,
        diagnosis=diagnosis,
    )
    _validate_recovery_chunk_policy(candidate)


def test_stage2_prompt_preserves_authoritative_unit_scale_fallback() -> None:
    assert (
        "authoritative full-task instruction with unit scales"
        in PROPOSAL_SYSTEM_PROMPT
    )
    assert "five-action replanning" in PROPOSAL_SYSTEM_PROMPT
    assert "remaining episode horizon" in PROPOSAL_SYSTEM_PROMPT
    assert "cooldown to cover the" in PROPOSAL_SYSTEM_PROMPT
    assert "maximum physical-action window" in PROPOSAL_SYSTEM_PROMPT
    assert "explicit non-reentry guard" in PROPOSAL_SYSTEM_PROMPT


def test_stage_recovers_completed_attempt_after_validator_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = CodexStageAgent(
        output_root=tmp_path,
        model="gpt-5.6-sol",
        reasoning_effort="high",
    )
    stage_root = tmp_path / "stage2-proposal"
    attempt = stage_root / "attempt-000"
    attempt.mkdir(parents=True)
    payload = {"objective": "one atomic candidate"}
    recovered = {"accepted": True}
    atomic_write_json(attempt / "output.json", recovered, overwrite=False)
    atomic_write_json(
        attempt / "invocation.json",
        {
            "session_id": agent.session_id,
            "provider_thread_id": "thread-recovered",
            "model": "gpt-5.6-sol",
            "reasoning_effort": "high",
            "provider_reported_resumed": True,
            "reconstructed": False,
            "evidence_policy": agent.EVIDENCE_POLICY,
            "error": None,
        },
        overwrite=False,
    )

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("provider must not be called for a recoverable attempt")

    monkeypatch.setattr("zetta.evolution.stages.build_planner", fail_if_called)
    def validate(value: dict[str, Any]) -> None:
        if value.get("accepted") is not True:
            raise ValueError("not accepted")

    result = agent._invoke(
        stage="stage2-proposal",
        system_prompt="system",
        payload=payload,
        validator=validate,
    )

    assert result == recovered
    context = read_json(stage_root / "context.json")
    assert context == {
        "session_id": agent.session_id,
        "provider_thread_id": "thread-recovered",
        "stage": "stage2-proposal",
        "model": "gpt-5.6-sol",
        "reasoning_effort": "high",
        "resumed": True,
        "reconstructed": False,
        "evidence_policy": agent.EVIDENCE_POLICY,
        "successful_attempt": "attempt-000",
        "output_sha256": canonical_sha256(recovered),
        "revalidated_completed_attempt": True,
    }


def test_stage_exposes_only_audited_evidence_capabilities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    class FakePlanner:
        def solve(self, **kwargs: Any) -> PlannerResult:
            captured["tool_names"] = {
                str(spec["name"])
                for spec in kwargs["toolkit"].get_tools_spec()
            }
            return PlannerResult(
                messages=[{"content": json.dumps({"accepted": True})}],
                stats={"thread_id": "thread-evidence-only"},
            )

    def build(kind: str, **kwargs: Any) -> FakePlanner:
        assert kind == "codex"
        captured["build"] = kwargs
        return FakePlanner()

    monkeypatch.setattr("zetta.evolution.stages.build_planner", build)
    root = tmp_path / "candidate"
    result = CodexStageAgent(
        output_root=root,
        artifact_reader=lambda _content_id: {"kind": "structured"},
    )._invoke(
        stage="evidence-only",
        system_prompt="system",
        payload={"bounded": True},
        validator=lambda value: None,
    )

    assert result == {"accepted": True}
    assert captured["tool_names"] == {
        "describe_tools",
        "read_campaign_artifact",
    }
    build_kwargs = captured["build"]
    assert build_kwargs["codex_repo_root"] == (
        root / "evidence-only" / "attempt-000"
    )
    assert build_kwargs["codex_sandbox"] == "read-only"
    assert build_kwargs["codex_native_tools"] is False


def test_stage_rejects_context_that_predates_evidence_isolation(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage2-proposal"
    stage.mkdir()
    atomic_write_json(stage / "input.json", {"bounded": True}, overwrite=False)
    atomic_write_json(stage / "output.json", {"accepted": True}, overwrite=False)
    atomic_write_json(
        stage / "context.json",
        {
            "session_id": "legacy-session",
            "provider_thread_id": "legacy-thread",
            "stage": "stage2-proposal",
            "successful_attempt": "attempt-000",
        },
        overwrite=False,
    )

    with pytest.raises(ValueError, match="predates enforced evidence isolation"):
        CodexStageAgent(output_root=tmp_path)._invoke(
            stage="stage2-proposal",
            system_prompt="system",
            payload={"bounded": True},
        )


def _diagnosis() -> CausalDiagnosis:
    payload = _diagnosis_payload()
    return CausalDiagnosis(
        **{
            **payload,
            "contributing_causes": tuple(payload["contributing_causes"]),
            "competing_hypotheses": tuple(payload["competing_hypotheses"]),
            "supporting_evidence_ids": tuple(payload["supporting_evidence_ids"]),
            "counterevidence_ids": tuple(payload["counterevidence_ids"]),
        }
    )


def test_stage_json_extractor_returns_outer_candidate_not_nested_parameters() -> None:
    payload = _candidate_payload()
    assert _extract_json_object("prefix\n" + json.dumps(payload))["candidate_id"] == (
        "candidate-stall"
    )


def test_stage1_task_contract_is_echoed_and_validated(tmp_path: Path, monkeypatch) -> None:
    contract = {
        "suite": "libero_10_swap",
        "task": "libero_10_swap/task6",
        "language": "Put the white mug on the plate and put the chocolate pudding to the right of the plate",
    }

    class FakePlanner:
        def solve(self, **_: Any) -> PlannerResult:
            payload = _diagnosis_payload()
            payload["task_contract"] = dict(contract)
            return PlannerResult(
                messages=[{"content": json.dumps(payload)}],
                stats={"thread_id": "thread-contract", "thread_resumed": False},
            )

    monkeypatch.setattr(
        "zetta.evolution.stages.build_planner", lambda *_args, **_kwargs: FakePlanner()
    )
    diagnosis = CodexStageAgent(output_root=tmp_path).diagnose(
        cluster=_cluster(),
        artifact_index=_artifact_index(),
        tool_catalog={"tools": []},
        task_contract=contract,
    )
    assert diagnosis.task_contract == contract
    assert _normalize_task_contract({**contract, "language": "  " + contract["language"]}) == contract


def test_stage1_rejects_task_contract_drift(tmp_path: Path, monkeypatch) -> None:
    expected = {
        "suite": "libero_10_swap",
        "task": "libero_10_swap/task6",
        "language": "Put the white mug on the plate",
    }

    class FakePlanner:
        def solve(self, **_: Any) -> PlannerResult:
            payload = _diagnosis_payload()
            payload["task_contract"] = {**expected, "language": "wrong object"}
            return PlannerResult(
                messages=[{"content": json.dumps(payload)}],
                stats={"thread_id": "thread-contract-drift", "thread_resumed": False},
            )

    monkeypatch.setattr(
        "zetta.evolution.stages.build_planner", lambda *_args, **_kwargs: FakePlanner()
    )
    with pytest.raises(ValueError, match="authoritative task contract"):
        CodexStageAgent(output_root=tmp_path).diagnose(
            cluster=_cluster(),
            artifact_index=_artifact_index(),
            tool_catalog={"tools": []},
            task_contract=expected,
        )


def test_stage2_resumes_the_exact_stage1_provider_thread(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[str | None] = []

    class FakePlanner:
        def __init__(self, payload: dict[str, Any], thread_id: str, resumed: bool):
            self.payload = payload
            self.thread_id = thread_id
            self.resumed = resumed

        def solve(self, **_: Any) -> PlannerResult:
            return PlannerResult(
                messages=[{"content": json.dumps(self.payload)}],
                stats={
                    "thread_id": self.thread_id,
                    "thread_resumed": self.resumed,
                },
            )

    def build(_kind: str, **kwargs: Any) -> FakePlanner:
        resume = kwargs.get("codex_thread_id")
        calls.append(resume)
        if resume is None:
            return FakePlanner(_diagnosis_payload(), "thread-stage-1", False)
        return FakePlanner(_candidate_payload(), "thread-stage-1", True)

    monkeypatch.setattr("zetta.evolution.stages.build_planner", build)
    stage1_root = tmp_path / "diagnosis"
    first = CodexStageAgent(output_root=stage1_root)
    diagnosis = first.diagnose(
        cluster=_cluster(), artifact_index=_artifact_index(), tool_catalog={"tools": []}
    )
    context = CodexStageAgent.load_context(stage1_root, "stage1-diagnosis")
    second = CodexStageAgent(
        output_root=tmp_path / "candidate",
        session_id=context["session_id"],
        thread_id=context["provider_thread_id"],
    )
    candidate = second.propose(
        generation=0,
        parent_sha256=None,
        diagnosis=diagnosis,
        tool_catalog={"tools": [{"name": "robocasa.vla.groot"}]},
        artifact_index=_artifact_index(),
        available_critic_features=("privileged.progress",),
    )

    assert candidate.candidate_id == "candidate-stall"
    assert calls == [None, "thread-stage-1"]
    stage2_context = CodexStageAgent.load_context(
        tmp_path / "candidate", "stage2-proposal"
    )
    assert stage2_context["session_id"] == context["session_id"]
    assert stage2_context["provider_thread_id"] == "thread-stage-1"
    assert stage2_context["resumed"] is True
    assert stage2_context["reconstructed"] is False


def test_failed_stage_can_append_a_changed_recovery_input(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "candidate"
    stage = root / "stage2-proposal"
    atomic_write_json(stage / "input.json", {"oversized": True}, overwrite=False)
    attempt = stage / "attempt-000"
    atomic_write_json(
        attempt / "invocation.json",
        {"error": "context window exceeded"},
        overwrite=False,
    )

    class FakePlanner:
        def solve(self, **_: Any) -> PlannerResult:
            return PlannerResult(
                messages=[{"content": json.dumps({"accepted": True})}],
                stats={"thread_id": "thread-recovery-input"},
            )

    monkeypatch.setattr(
        "zetta.evolution.stages.build_planner", lambda *_args, **_kwargs: FakePlanner()
    )
    result = CodexStageAgent(output_root=root)._invoke(
        stage="stage2-proposal",
        system_prompt="system",
        payload={"bounded": True},
        validator=lambda value: None,
    )

    assert result == {"accepted": True}
    assert read_json(stage / "input.json") == {"oversized": True}
    recovery_input = stage / "input-recovery-000.json"
    assert read_json(recovery_input) == {"bounded": True}
    context = read_json(stage / "context.json")
    assert context["successful_attempt"] == "attempt-001"
    assert context["input_artifact"] == recovery_input.name
    assert context["input_sha256"] == canonical_sha256({"bounded": True})
    assert context["append_only_recovery_input"] is True


def test_stage2_rejects_unobserved_critic_feature_before_commit(
    tmp_path: Path, monkeypatch
) -> None:
    class FakePlanner:
        def solve(self, **_: Any) -> PlannerResult:
            return PlannerResult(
                messages=[{"content": json.dumps(_candidate_payload())}],
                stats={"thread_id": "thread-feature-check"},
            )

    monkeypatch.setattr(
        "zetta.evolution.stages.build_planner", lambda *_args, **_kwargs: FakePlanner()
    )
    root = tmp_path / "candidate"
    with pytest.raises(ValueError, match="were not observed"):
        CodexStageAgent(output_root=root).propose(
            generation=0,
            parent_sha256=None,
            diagnosis=_diagnosis(),
            tool_catalog={"tools": [{"name": "robocasa.vla.groot"}]},
            artifact_index=_artifact_index(),
            available_critic_features=("privileged.other",),
        )
    assert not (root / "stage2-proposal" / "output.json").exists()


def test_stage2_receives_only_prompt_safe_provisional_authorization(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, Any] = {}

    class FakePlanner:
        def solve(self, **kwargs: Any) -> PlannerResult:
            captured.update(json.loads(kwargs["user_message"]))
            return PlannerResult(
                messages=[{"content": json.dumps(_candidate_payload())}],
                stats={"thread_id": "thread-provisional"},
            )

    monkeypatch.setattr(
        "zetta.evolution.stages.build_planner", lambda *_args, **_kwargs: FakePlanner()
    )
    authorization = {
        "authorization_id": "provisional-123",
        "authorization_kind": "timeboxed_provisional_hypothesis_test",
        "diagnosis_sha256": _diagnosis().sha256,
        "cluster_target_sha256": "a" * 64,
        "minimum_same_seed_successes": 1,
        "same_seed_pass_rate": 0.25,
        "skip_regression": True,
        "heldout_label": "fixed_20_seed_validation_sr_not_unbiased_test",
        "deadline": "2026-08-08T15:50:00+08:00",
        "diagnosis_remains_inconclusive": True,
        "episodes_ledger_sha256": "must-not-enter-prompt",
    }

    CodexStageAgent(output_root=tmp_path / "candidate").propose(
        generation=0,
        parent_sha256=None,
        diagnosis=_diagnosis(),
        tool_catalog={"tools": [{"name": "robocasa.vla.groot"}]},
        artifact_index=_artifact_index(),
        available_critic_features=("privileged.progress",),
        provisional_hypothesis=authorization,
    )

    prompt_value = captured["provisional_hypothesis_authorization"]
    assert prompt_value["authorization_id"] == "provisional-123"
    assert prompt_value["diagnosis_remains_inconclusive"] is True
    assert "episodes_ledger_sha256" not in prompt_value


def test_formal_stage_attests_sol_high_in_prompt_and_audit(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, Any] = {}

    class FakePlanner:
        def solve(self, **kwargs: Any) -> PlannerResult:
            captured.update(kwargs)
            return PlannerResult(
                messages=[{"content": json.dumps(_diagnosis_payload())}],
                stats={
                    "thread_id": "thread-sol-high",
                    "thread_resumed": False,
                    "model": "gpt-5.6-sol",
                    "reasoning_effort": "high",
                },
            )

    def build(_kind: str, **kwargs: Any) -> FakePlanner:
        assert kwargs["model"] == "gpt-5.6-sol"
        assert kwargs["reasoning_effort"] == "high"
        return FakePlanner()

    monkeypatch.setattr("zetta.evolution.stages.build_planner", build)
    root = tmp_path / "formal-stage"
    CodexStageAgent(
        output_root=root,
        model="gpt-5.6-sol",
        reasoning_effort="high",
    ).diagnose(
        cluster=_cluster(), artifact_index=_artifact_index(), tool_catalog={"tools": []}
    )

    model_input = json.loads(captured["user_message"])
    assert model_input["agent_contract"] == {
        "model": "gpt-5.6-sol",
        "reasoning_effort": "high",
    }
    invocation = next((root / "stage1-diagnosis").glob("attempt-*/invocation.json"))
    audit = read_json(invocation)
    assert audit["model"] == "gpt-5.6-sol"
    assert audit["reasoning_effort"] == "high"


def test_stage2_rejects_an_unchanged_failed_gate_mechanism(
    tmp_path: Path, monkeypatch
) -> None:
    previous = _candidate_payload()
    payload = _candidate_payload()
    payload["candidate_id"] = "candidate-renamed-only"
    payload["critic_rules"][0]["title"] = "renamed critic only"
    payload["recovery_rules"][0]["title"] = "renamed recovery only"

    class FakePlanner:
        def solve(self, **_: Any) -> PlannerResult:
            return PlannerResult(
                messages=[{"content": json.dumps(payload)}],
                stats={"thread_id": "thread-refinement"},
            )

    monkeypatch.setattr(
        "zetta.evolution.stages.build_planner", lambda *_args, **_kwargs: FakePlanner()
    )
    root = tmp_path / "candidate"
    with pytest.raises(ValueError, match="rejected mechanism unchanged"):
        CodexStageAgent(output_root=root).propose(
            generation=0,
            parent_sha256=None,
            diagnosis=_diagnosis(),
            tool_catalog={"tools": [{"name": "robocasa.vla.groot"}]},
            artifact_index=_artifact_index(),
            available_critic_features=("privileged.progress",),
            refinement_context={
                "mode": "refine_rejected_candidate",
                "previous_candidate": previous,
                "paired_gate_result": {
                    "passed": False,
                    "candidate_successes": 0,
                    "parent_successes": 0,
                    "paired_count": 1,
                },
                "gate_evidence": _artifact_index()["artifacts"],
            },
        )
    stage = root / "stage2-proposal"
    assert (stage / "attempt-000" / "output.json").is_file()
    assert not (stage / "output.json").exists()


def test_stage2_refinement_must_cite_paired_gate_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    payload = _candidate_payload()
    payload["critic_rules"][0]["threshold"] = 0.02
    gate_id = "artifact-" + "3" * 64
    index = _artifact_index()
    index["artifacts"].append(
        {
            "content_id": gate_id,
            "type": "candidate_gate_episode",
            "summary": "valid unsuccessful candidate gate arm evidence",
            "hash": "hmac-sha256:" + "4" * 64,
        }
    )

    class FakePlanner:
        def solve(self, **_: Any) -> PlannerResult:
            return PlannerResult(
                messages=[{"content": json.dumps(payload)}],
                stats={"thread_id": "thread-gate-evidence"},
            )

    monkeypatch.setattr(
        "zetta.evolution.stages.build_planner", lambda *_args, **_kwargs: FakePlanner()
    )
    with pytest.raises(ValueError, match="did not cite rejected-gate evidence"):
        CodexStageAgent(output_root=tmp_path / "candidate").propose(
            generation=0,
            parent_sha256=None,
            diagnosis=_diagnosis(),
            tool_catalog={"tools": [{"name": "robocasa.vla.groot"}]},
            artifact_index=index,
            available_critic_features=("privileged.progress",),
            refinement_context={
                "mode": "refine_rejected_candidate",
                "previous_candidate": _candidate_payload(),
                "paired_gate_result": {"passed": False, "paired_count": 1},
                "gate_evidence": [index["artifacts"][-1]],
            },
        )


def test_stage2_refinement_may_retain_previous_candidate_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    payload = _candidate_payload()
    payload["critic_rules"][0]["threshold"] = 0.02
    inherited_id = "artifact-" + "7" * 64
    payload["critic_rules"][0]["evidence_ids"] = [inherited_id]
    previous = _candidate_payload()
    previous["critic_rules"][0]["evidence_ids"] = [inherited_id]

    class FakePlanner:
        def solve(self, **_: Any) -> PlannerResult:
            return PlannerResult(
                messages=[{"content": json.dumps(payload)}],
                stats={"thread_id": "thread-inherited-evidence"},
            )

    monkeypatch.setattr(
        "zetta.evolution.stages.build_planner", lambda *_args, **_kwargs: FakePlanner()
    )
    candidate = CodexStageAgent(output_root=tmp_path / "candidate").propose(
        generation=0,
        parent_sha256=None,
        diagnosis=_diagnosis(),
        tool_catalog={"tools": [{"name": "robocasa.vla.groot"}]},
        artifact_index=_artifact_index(),
        available_critic_features=("privileged.progress",),
        refinement_context={
            "mode": "refine_rejected_candidate",
            "previous_candidate": previous,
            "paired_gate_result": {"passed": False, "paired_count": 1},
            "gate_evidence": _artifact_index()["artifacts"],
        },
    )
    assert candidate.critic_rules[0].evidence_ids == (inherited_id,)


def test_missing_provider_thread_is_reconstructed_append_only(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[str | None] = []

    class FakePlanner:
        def __init__(self, resume: str | None):
            self.resume = resume

        def solve(self, **_: Any) -> PlannerResult:
            if self.resume is not None:
                return PlannerResult(
                    error="provider thread is unavailable",
                    stats={"thread_id": self.resume, "thread_resumed": True},
                )
            return PlannerResult(
                messages=[{"content": json.dumps(_candidate_payload())}],
                stats={"thread_id": "thread-rebuilt", "thread_resumed": False},
            )

    def build(_kind: str, **kwargs: Any) -> FakePlanner:
        resume = kwargs.get("codex_thread_id")
        calls.append(resume)
        return FakePlanner(resume)

    monkeypatch.setattr("zetta.evolution.stages.build_planner", build)
    root = tmp_path / "candidate"
    agent = CodexStageAgent(
        output_root=root,
        session_id="logical-session",
        thread_id="thread-missing",
    )
    agent.propose(
        generation=0,
        parent_sha256=None,
        diagnosis=_diagnosis(),
        tool_catalog={"tools": [{"name": "robocasa.vla.groot"}]},
        artifact_index=_artifact_index(),
    )

    assert calls == ["thread-missing", None]
    stage = root / "stage2-proposal"
    assert (stage / "attempt-000" / "invocation.json").is_file()
    assert (stage / "attempt-001" / "invocation.json").is_file()
    assert read_json(stage / "context.json")["reconstructed"] is True
    assert read_json(stage / "context.json")["provider_thread_id"] == "thread-rebuilt"


def test_semantically_invalid_stage_output_is_audited_but_not_committed(
    tmp_path: Path, monkeypatch
) -> None:
    invalid = _diagnosis_payload()
    invalid["supporting_evidence_ids"] = ["artifact-" + "f" * 64]

    class FakePlanner:
        def solve(self, **_: Any) -> PlannerResult:
            return PlannerResult(
                messages=[{"content": json.dumps(invalid)}],
                stats={"thread_id": "thread-invalid", "thread_resumed": False},
            )

    monkeypatch.setattr(
        "zetta.evolution.stages.build_planner", lambda *_args, **_kwargs: FakePlanner()
    )
    root = tmp_path / "diagnosis"
    agent = CodexStageAgent(output_root=root)
    with pytest.raises(ValueError, match="unknown evidence"):
        agent.diagnose(
            cluster=_cluster(),
            artifact_index=_artifact_index(),
            tool_catalog={"tools": []},
        )
    stage = root / "stage1-diagnosis"
    assert (stage / "attempt-000" / "output.json").is_file()
    assert not (stage / "output.json").exists()


def test_committed_output_recovers_context_without_reinvoking_provider(
    tmp_path: Path, monkeypatch
) -> None:
    calls = 0

    class FakePlanner:
        def solve(self, **_: Any) -> PlannerResult:
            nonlocal calls
            calls += 1
            return PlannerResult(
                messages=[{"content": json.dumps(_diagnosis_payload())}],
                stats={"thread_id": "thread-recover", "thread_resumed": False},
            )

    monkeypatch.setattr(
        "zetta.evolution.stages.build_planner", lambda *_args, **_kwargs: FakePlanner()
    )
    root = tmp_path / "diagnosis"
    arguments = {
        "cluster": _cluster(),
        "artifact_index": _artifact_index(),
        "tool_catalog": {"tools": []},
    }
    CodexStageAgent(output_root=root).diagnose(**arguments)
    context_path = root / "stage1-diagnosis" / "context.json"
    context_path.unlink()

    recovered = CodexStageAgent(output_root=root).diagnose(**arguments)
    context = read_json(context_path)
    assert recovered.diagnosis_id == "diagnosis-stall"
    assert calls == 1
    assert context["provider_thread_id"] == "thread-recover"
    assert context["context_recovered_after_commit"] is True
