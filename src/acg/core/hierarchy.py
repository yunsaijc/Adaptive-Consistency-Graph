from __future__ import annotations

import math
from dataclasses import replace
from hashlib import sha1
from types import MappingProxyType
from typing import Mapping, Protocol

from .relations import EdgeKey, hierarchy_edge_weights
from .schema import HierarchyCluster, MemoryHierarchy
from .store import MemoryGraph


class HierarchyBuilder(Protocol):
    def build(self, graph: MemoryGraph) -> MemoryHierarchy:
        raise NotImplementedError


class FlatHierarchyBuilder:
    def build(self, graph: MemoryGraph) -> MemoryHierarchy:
        root = HierarchyCluster(
            id="cluster-root", parent_id=None, member_ids=tuple(sorted(graph.units))
        )
        return MemoryHierarchy(
            root_id=root.id,
            clusters=MappingProxyType({root.id: root}),
            entropy_bits=one_dimensional_entropy(
                graph, edge_weights=hierarchy_edge_weights(graph)
            ),
        )


class StructuralEntropyHierarchyBuilder:
    def __init__(self, *, tolerance: float = 1e-12) -> None:
        if tolerance < 0.0:
            raise ValueError("Structural entropy tolerance must be non-negative")
        self.tolerance = tolerance

    def build(self, graph: MemoryGraph) -> MemoryHierarchy:
        edge_weights = hierarchy_edge_weights(graph)
        all_members = frozenset(graph.units)
        root = HierarchyCluster(
            id="cluster-root", parent_id=None, member_ids=tuple(sorted(all_members))
        )
        clusters: dict[str, HierarchyCluster] = {root.id: root}
        root_partition, root_entropy = self._best_partition(
            graph, all_members, edge_weights
        )
        if _is_nontrivial(root_partition, all_members):
            self._add_partition(
                graph,
                edge_weights,
                parent_id=root.id,
                partition=root_partition,
                clusters=clusters,
            )
        return MemoryHierarchy(
            root_id=root.id,
            clusters=MappingProxyType(clusters),
            entropy_bits=root_entropy,
        )

    def _add_partition(
        self,
        graph: MemoryGraph,
        edge_weights: Mapping[EdgeKey, float],
        *,
        parent_id: str,
        partition: tuple[frozenset[str], ...],
        clusters: dict[str, HierarchyCluster],
    ) -> None:
        child_ids: list[str] = []
        for members in sorted(partition, key=lambda item: tuple(sorted(item))):
            cluster_id = _cluster_id(members)
            child_ids.append(cluster_id)
            clusters[cluster_id] = HierarchyCluster(
                id=cluster_id, parent_id=parent_id, member_ids=tuple(sorted(members))
            )
            if len(members) <= 2:
                continue
            child_partition, _entropy = self._best_partition(
                graph, members, edge_weights
            )
            if _is_nontrivial(child_partition, members):
                self._add_partition(
                    graph,
                    edge_weights,
                    parent_id=cluster_id,
                    partition=child_partition,
                    clusters=clusters,
                )
        clusters[parent_id] = replace(clusters[parent_id], child_ids=tuple(child_ids))

    def _best_partition(
        self,
        graph: MemoryGraph,
        members: frozenset[str],
        edge_weights: Mapping[EdgeKey, float],
    ) -> tuple[tuple[frozenset[str], ...], float]:
        partition = tuple((frozenset((memory_id,)) for memory_id in sorted(members)))
        current_entropy = partition_structural_entropy(
            graph, partition, members=members, edge_weights=edge_weights
        )
        while len(partition) > 1:
            best_partition: tuple[frozenset[str], ...] | None = None
            best_entropy = current_entropy
            for left_index in range(len(partition)):
                for right_index in range(left_index + 1, len(partition)):
                    left = partition[left_index]
                    right = partition[right_index]
                    if _cross_weight(edge_weights, left, right) <= 0.0:
                        continue
                    merged = left | right
                    candidate = tuple(
                        (
                            module
                            for index, module in enumerate(partition)
                            if index not in {left_index, right_index}
                        )
                    ) + (merged,)
                    candidate = tuple(
                        sorted(candidate, key=lambda item: tuple(sorted(item)))
                    )
                    entropy = partition_structural_entropy(
                        graph, candidate, members=members, edge_weights=edge_weights
                    )
                    if entropy < best_entropy - self.tolerance:
                        best_entropy = entropy
                        best_partition = candidate
            if best_partition is None:
                break
            partition = best_partition
            current_entropy = best_entropy
        return (partition, current_entropy)


def one_dimensional_entropy(
    graph: MemoryGraph,
    *,
    members: frozenset[str] | None = None,
    edge_weights: Mapping[EdgeKey, float] | None = None,
) -> float:
    node_ids = frozenset(graph.units) if members is None else members
    projected = _relation_edge_weights(graph) if edge_weights is None else edge_weights
    degrees, total_volume = _degrees_and_volume(graph, node_ids, projected)
    if total_volume <= 0.0:
        return 0.0
    return -sum(
        (
            degree / total_volume * math.log2(degree / total_volume)
            for degree in degrees.values()
            if degree > 0.0
        )
    )


def partition_structural_entropy(
    graph: MemoryGraph,
    partition: tuple[frozenset[str], ...],
    *,
    members: frozenset[str] | None = None,
    edge_weights: Mapping[EdgeKey, float] | None = None,
) -> float:
    node_ids = frozenset(graph.units) if members is None else members
    _validate_partition(partition, node_ids)
    projected = _relation_edge_weights(graph) if edge_weights is None else edge_weights
    degrees, total_volume = _degrees_and_volume(graph, node_ids, projected)
    if total_volume <= 0.0:
        return 0.0
    entropy = 0.0
    for module in partition:
        volume = sum((degrees[memory_id] for memory_id in module))
        if volume <= 0.0:
            continue
        boundary = _boundary_weight(projected, module, node_ids)
        if boundary > 0.0:
            entropy -= boundary / total_volume * math.log2(volume / total_volume)
        entropy -= sum(
            (
                degrees[memory_id]
                / total_volume
                * math.log2(degrees[memory_id] / volume)
                for memory_id in module
                if degrees[memory_id] > 0.0
            )
        )
    return entropy


def _degrees_and_volume(
    graph: MemoryGraph, members: frozenset[str], edge_weights: Mapping[EdgeKey, float]
) -> tuple[dict[str, float], float]:
    unknown = members - set(graph.units)
    if unknown:
        raise ValueError(
            f"Unknown memories in structural entropy graph: {sorted(unknown)}"
        )
    degrees = {memory_id: 0.0 for memory_id in members}
    for (left, right), weight in edge_weights.items():
        if left not in members or right not in members:
            continue
        degrees[left] += weight
        degrees[right] += weight
    return (degrees, sum(degrees.values()))


def _boundary_weight(
    edge_weights: Mapping[EdgeKey, float],
    module: frozenset[str],
    members: frozenset[str],
) -> float:
    return sum(
        (
            weight
            for (left, right), weight in edge_weights.items()
            if left in members
            and right in members
            and ((left in module) != (right in module))
        )
    )


def _cross_weight(
    edge_weights: Mapping[EdgeKey, float], left: frozenset[str], right: frozenset[str]
) -> float:
    return sum(
        (
            weight
            for (source, target), weight in edge_weights.items()
            if source in left
            and target in right
            or (source in right and target in left)
        )
    )


def _relation_edge_weights(graph: MemoryGraph) -> dict[EdgeKey, float]:
    return {
        (relation.source, relation.target): relation.weight
        for relation in graph.relations.values()
    }


def _validate_partition(
    partition: tuple[frozenset[str], ...], members: frozenset[str]
) -> None:
    if not partition and members:
        raise ValueError("Structural entropy partition must not be empty")
    seen: set[str] = set()
    for module in partition:
        if not module:
            raise ValueError("Structural entropy module must not be empty")
        overlap = seen.intersection(module)
        if overlap:
            raise ValueError(f"Structural entropy modules overlap: {sorted(overlap)}")
        seen.update(module)
    if seen != set(members):
        raise ValueError(
            f"Structural entropy partition must exactly cover its graph: missing={sorted(set(members) - seen)}, unknown={sorted(seen - set(members))}"
        )


def _is_nontrivial(
    partition: tuple[frozenset[str], ...], members: frozenset[str]
) -> bool:
    return 1 < len(partition) < len(members)


def _cluster_id(members: frozenset[str]) -> str:
    digest = sha1("\x00".join(sorted(members)).encode("utf-8")).hexdigest()[:12]
    return f"cluster-{digest}"
