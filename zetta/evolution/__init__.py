# Copyright (c) 2026 Zetta Contributors
"""Auditable rollout evolution for critic and recovery bundles.

This package is intentionally independent from :mod:`zetta.memory`.  Episode
planners consume one frozen candidate bundle; only the campaign coordinator
may promote a new bundle after every configured gate has passed.
"""

from zetta.evolution.models import (
    CampaignManifest,
    CampaignPhase,
    CandidateBundle,
    CausalDiagnosis,
    CriticPredicate,
    CriticRule,
    EpisodeRecord,
    FailureCluster,
    FailureSegment,
    GateDecision,
    RecoveryRule,
    TrajectoryIndex,
)

__all__ = [
    "CampaignManifest",
    "CampaignPhase",
    "CandidateBundle",
    "CausalDiagnosis",
    "CriticPredicate",
    "CriticRule",
    "EpisodeRecord",
    "TrajectoryIndex",
    "FailureCluster",
    "FailureSegment",
    "GateDecision",
    "RecoveryRule",
]
