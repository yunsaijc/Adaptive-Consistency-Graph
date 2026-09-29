from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from .store import MemoryGraph

EdgeKey = tuple[str, str]


def rebuild_semantic_relations(graph: MemoryGraph) -> tuple[str, ...]:
    return graph.replace_relations(mutual_nearest_edge_weights(graph))


def mutual_nearest_edge_weights(graph: MemoryGraph) -> dict[EdgeKey, float]:
    vectors = {
        memory_id: unit.embedding
        for memory_id, unit in graph.units.items()
        if unit.embedding is not None
    }
    nearest: dict[str, str] = {}
    for memory_id, vector in vectors.items():
        candidates = [
            (cosine_similarity(vector, other_vector), other_id)
            for other_id, other_vector in vectors.items()
            if other_id != memory_id
        ]
        if candidates:
            nearest[memory_id] = min(candidates, key=lambda item: (-item[0], item[1]))[
                1
            ]
    return {
        _edge_key(left, right): 1.0
        for left, right in nearest.items()
        if nearest.get(right) == left and left < right
    }


def source_affinity_edge_weights(graph: MemoryGraph) -> dict[EdgeKey, float]:
    members_by_source: dict[str, set[str]] = defaultdict(set)
    for memory_id, unit in graph.units.items():
        for source_id in unit.source_refs:
            members_by_source[source_id].add(memory_id)
    unique_groups = {
        tuple(sorted(members))
        for members in members_by_source.values()
        if len(members) > 1
    }
    edges: dict[EdgeKey, float] = defaultdict(float)
    for members in sorted(unique_groups):
        contribution = 1.0 / (len(members) - 1)
        for left_index, left in enumerate(members):
            for right in members[left_index + 1 :]:
                edges[_edge_key(left, right)] += contribution
    return dict(edges)


def hierarchy_edge_weights(graph: MemoryGraph) -> dict[EdgeKey, float]:
    layers = (
        {
            _edge_key(relation.source, relation.target): relation.weight
            for relation in graph.relations.values()
        },
        source_affinity_edge_weights(graph),
    )
    merged: dict[EdgeKey, float] = defaultdict(float)
    for layer in layers:
        for edge, weight in layer.items():
            merged[edge] += weight
    return dict(merged)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum((a * b for a, b in zip(left, right)))
    left_norm = math.sqrt(sum((value * value for value in left)))
    right_norm = math.sqrt(sum((value * value for value in right)))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _edge_key(left: str, right: str) -> EdgeKey:
    if left == right:
        raise ValueError("Semantic relation cannot be a self-edge")
    return (left, right) if left < right else (right, left)
