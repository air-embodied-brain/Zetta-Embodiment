# Copyright (c) 2026 RPent Contributors
"""Deterministic failure segmentation clustering and medoid selection."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable

from rpent.evolution.models import FailureCluster, FailureSegment

TOKEN_PATTERN = re.compile(r"[a-z0-9_\-]+", re.IGNORECASE)


def _tokens(text: str) -> Counter[str]:
    return Counter(token.lower() for token in TOKEN_PATTERN.findall(text))


def _counter_cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _vector_cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float | None:
    if not left or len(left) != len(right):
        return None
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return None
    return dot / (left_norm * right_norm)


def segment_similarity(left: FailureSegment, right: FailureSegment) -> float:
    text_similarity = _counter_cosine(_tokens(left.summary), _tokens(right.summary))
    embedding_similarity = _vector_cosine(left.embedding, right.embedding)
    state_similarity = _vector_cosine(left.state_signature, right.state_signature)
    weighted = [(text_similarity, 0.6)]
    if embedding_similarity is not None:
        weighted.append((max(0.0, embedding_similarity), 0.3))
    if state_similarity is not None:
        weighted.append((max(0.0, state_similarity), 0.1))
    total_weight = sum(weight for _, weight in weighted)
    return sum(value * weight for value, weight in weighted) / total_weight


def _complete_link_clusters(
    group: list[FailureSegment], threshold: float
) -> list[list[FailureSegment]]:
    """Deterministic agglomerative complete-link clustering.

    Complete-link prevents an A~B~C similarity chain from merging A and C when
    their own failure mechanisms are dissimilar.
    """

    clusters = [[item] for item in sorted(group, key=lambda row: row.segment_id)]
    while True:
        best: tuple[float, tuple[str, ...], int, int] | None = None
        for left_index, left in enumerate(clusters):
            for right_index in range(left_index + 1, len(clusters)):
                right = clusters[right_index]
                similarity = min(segment_similarity(a, b) for a in left for b in right)
                if similarity < threshold:
                    continue
                identifiers = tuple(sorted(item.segment_id for item in (*left, *right)))
                candidate = (similarity, identifiers, left_index, right_index)
                if best is None or (-candidate[0], candidate[1]) < (
                    -best[0],
                    best[1],
                ):
                    best = candidate
        if best is None:
            return clusters
        _, _, left_index, right_index = best
        merged = sorted(
            (*clusters[left_index], *clusters[right_index]),
            key=lambda row: row.segment_id,
        )
        clusters[left_index] = merged
        del clusters[right_index]


def cluster_failure_segments(
    segments: Iterable[FailureSegment],
    *,
    similarity_threshold: float = 0.72,
    representatives: int = 3,
) -> list[FailureCluster]:
    """Cluster within hard failure/stage/tool groups, then rank by prevalence."""

    if not 0 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be in [0, 1]")
    ordered = sorted(segments, key=lambda item: item.segment_id)
    if not ordered:
        return []
    if len({item.segment_id for item in ordered}) != len(ordered):
        raise ValueError("failure segment identifiers must be unique")
    hard_groups: dict[tuple[str, str, str], list[FailureSegment]] = defaultdict(list)
    for segment in ordered:
        key = (segment.failure_class, segment.stage, segment.tool or "")
        hard_groups[key].append(segment)

    raw_clusters: list[tuple[tuple[str, str, str], list[FailureSegment]]] = []
    for hard_key, group in sorted(hard_groups.items()):
        for component in _complete_link_clusters(group, similarity_threshold):
            raw_clusters.append((hard_key, component))

    result: list[FailureCluster] = []
    total = len(ordered)
    for index, (hard_key, members) in enumerate(
        sorted(
            raw_clusters,
            key=lambda item: (
                -len(item[1]),
                -sum(segment.severity for segment in item[1]),
                item[0],
                item[1][0].segment_id,
            ),
        ),
        start=1,
    ):
        pair_scores = {
            member.segment_id: sum(
                segment_similarity(member, other) for other in members
            )
            for member in members
        }
        medoid = min(
            members,
            key=lambda member: (-pair_scores[member.segment_id], member.segment_id),
        )
        ranked_representatives = sorted(
            members,
            key=lambda member: (
                -member.severity,
                -pair_scores[member.segment_id],
                member.segment_id,
            ),
        )[: max(1, representatives)]
        result.append(
            FailureCluster(
                cluster_id=f"cluster-{index:03d}",
                hard_key=hard_key,
                member_segment_ids=tuple(member.segment_id for member in members),
                episode_ids=tuple(sorted({member.episode_id for member in members})),
                representative_segment_ids=tuple(
                    member.segment_id for member in ranked_representatives
                ),
                medoid_segment_id=medoid.segment_id,
                summary=medoid.summary,
                prevalence=len(members) / total,
                mean_severity=sum(member.severity for member in members) / len(members),
            )
        )
    return result
