from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .hierarchy import StructuralEntropyHierarchyBuilder, partition_structural_entropy
from .relations import EdgeKey
from .store import MemoryGraph


@dataclass(frozen=True)
class _Module:
    members: frozenset[str]
    volume: float
    boundary: float


class EfficientStructuralEntropyHierarchyBuilder(StructuralEntropyHierarchyBuilder):
    def _best_partition(
        self,
        graph: MemoryGraph,
        members: frozenset[str],
        edge_weights: Mapping[EdgeKey, float],
    ) -> tuple[tuple[frozenset[str], ...], float]:
        modules, cross_weights, total_volume = _initial_modules(
            graph, members, edge_weights
        )
        while len(modules) > 1 and cross_weights:
            ordered = sorted(modules, key=_module_key)
            order = {module_id: index for index, module_id in enumerate(ordered)}
            connected_pairs = sorted(
                cross_weights,
                key=lambda pair: (
                    min(order[pair[0]], order[pair[1]]),
                    max(order[pair[0]], order[pair[1]]),
                ),
            )
            best_pair: tuple[frozenset[str], frozenset[str]] | None = None
            best_delta = 0.0
            for raw_left, raw_right in connected_pairs:
                left, right = (
                    (raw_left, raw_right)
                    if order[raw_left] < order[raw_right]
                    else (raw_right, raw_left)
                )
                cross_weight = cross_weights[_edge_key(left, right)]
                delta = _merge_entropy_delta(
                    modules[left],
                    modules[right],
                    cross_weight=cross_weight,
                    total_volume=total_volume,
                )
                if delta < best_delta - self.tolerance:
                    best_delta = delta
                    best_pair = (left, right)
            if best_pair is None:
                break
            _merge_modules(modules, cross_weights, *best_pair)
        partition = tuple(sorted(modules, key=_module_key))
        entropy = partition_structural_entropy(
            graph, partition, members=members, edge_weights=edge_weights
        )
        return (partition, entropy)


def _initial_modules(
    graph: MemoryGraph, members: frozenset[str], edge_weights: Mapping[EdgeKey, float]
) -> tuple[
    dict[frozenset[str], _Module],
    dict[tuple[frozenset[str], frozenset[str]], float],
    float,
]:
    unknown = members - set(graph.units)
    if unknown:
        raise ValueError(
            f"Unknown memories in structural entropy graph: {sorted(unknown)}"
        )
    degrees = {memory_id: 0.0 for memory_id in members}
    cross_weights: dict[tuple[frozenset[str], frozenset[str]], float] = {}
    for (left, right), weight in edge_weights.items():
        if left not in members or right not in members:
            continue
        degrees[left] += weight
        degrees[right] += weight
        left_module = frozenset((left,))
        right_module = frozenset((right,))
        key = _edge_key(left_module, right_module)
        cross_weights[key] = cross_weights.get(key, 0.0) + weight
    modules = {
        frozenset((memory_id,)): _Module(
            members=frozenset((memory_id,)), volume=degree, boundary=degree
        )
        for memory_id, degree in degrees.items()
    }
    return (modules, cross_weights, sum(degrees.values()))


def _merge_entropy_delta(
    left: _Module, right: _Module, *, cross_weight: float, total_volume: float
) -> float:
    merged_volume = left.volume + right.volume
    merged_boundary = left.boundary + right.boundary - 2.0 * cross_weight
    return (
        _module_entropy_term(merged_volume, merged_boundary, total_volume)
        - _module_entropy_term(left.volume, left.boundary, total_volume)
        - _module_entropy_term(right.volume, right.boundary, total_volume)
    )


def _module_entropy_term(volume: float, boundary: float, total_volume: float) -> float:
    if volume <= 0.0 or total_volume <= 0.0:
        return 0.0
    term = volume / total_volume * math.log2(volume)
    if boundary > 0.0:
        term -= boundary / total_volume * math.log2(volume / total_volume)
    return term


def _merge_modules(
    modules: dict[frozenset[str], _Module],
    cross_weights: dict[tuple[frozenset[str], frozenset[str]], float],
    left_id: frozenset[str],
    right_id: frozenset[str],
) -> None:
    left = modules.pop(left_id)
    right = modules.pop(right_id)
    pair = _edge_key(left_id, right_id)
    cross_weight = cross_weights.pop(pair)
    merged_id = left_id | right_id
    merged = _Module(
        members=merged_id,
        volume=left.volume + right.volume,
        boundary=left.boundary + right.boundary - 2.0 * cross_weight,
    )
    for other_id in tuple(modules):
        left_key = _edge_key(left_id, other_id)
        right_key = _edge_key(right_id, other_id)
        combined = cross_weights.pop(left_key, 0.0) + cross_weights.pop(right_key, 0.0)
        if combined > 0.0:
            cross_weights[_edge_key(merged_id, other_id)] = combined
    modules[merged_id] = merged


def _module_key(members: frozenset[str]) -> tuple[str, ...]:
    return tuple(sorted(members))


def _edge_key(
    left: frozenset[str], right: frozenset[str]
) -> tuple[frozenset[str], frozenset[str]]:
    if _module_key(left) < _module_key(right):
        return (left, right)
    return (right, left)


__all__ = ["EfficientStructuralEntropyHierarchyBuilder"]
