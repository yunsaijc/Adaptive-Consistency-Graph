from __future__ import annotations

import math
from dataclasses import dataclass
from hashlib import sha1
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence

from .schema import MemoryHierarchy
from .store import MemoryGraph


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        raise NotImplementedError


@dataclass(frozen=True)
class RetrievedSubgraph:
    query: str
    seed_ids: tuple[str, ...]
    memory_ids: tuple[str, ...]
    relation_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    closure_groups: tuple[tuple[str, ...], ...]
    scores: Mapping[str, float]


def index_memories(
    graph: MemoryGraph,
    embedding_provider: EmbeddingProvider,
    memory_ids: Sequence[str] | None = None,
) -> tuple[str, ...]:
    selected = (
        tuple(graph.units) if memory_ids is None else tuple(dict.fromkeys(memory_ids))
    )
    existing = [memory_id for memory_id in selected if memory_id in graph.units]
    if not existing:
        return ()
    vectors = embedding_provider.embed(
        [graph.units[memory_id].content for memory_id in existing]
    )
    if len(vectors) != len(existing):
        raise RuntimeError("Embedding provider returned the wrong number of vectors")
    graph.set_embeddings(dict(zip(existing, vectors)))
    return tuple(existing)


def retrieve_memory(
    graph: MemoryGraph,
    *,
    query: str,
    embedding_provider: EmbeddingProvider,
    hierarchy: MemoryHierarchy | None = None,
    top_k: int = 8,
    closure_depth: int = 1,
    route_width: int = 2,
) -> RetrievedSubgraph:
    if top_k <= 0:
        raise ValueError("Retrieval top_k must be positive")
    if closure_depth < 0:
        raise ValueError("Closure depth must be non-negative")
    if route_width <= 0:
        raise ValueError("Hierarchy route width must be positive")
    normalized_query = query.strip()
    if not normalized_query or not graph.units:
        return RetrievedSubgraph(
            query=normalized_query,
            seed_ids=(),
            memory_ids=(),
            relation_ids=(),
            source_ids=(),
            closure_groups=(),
            scores=MappingProxyType({}),
        )
    missing = [
        memory_id for memory_id, unit in graph.units.items() if unit.embedding is None
    ]
    batch_texts = [
        normalized_query,
        *(graph.units[memory_id].content for memory_id in missing),
    ]
    vectors = embedding_provider.embed(batch_texts)
    if len(vectors) != len(batch_texts):
        raise RuntimeError("Embedding provider returned the wrong number of vectors")
    query_vector = _normalize_vector(vectors[0])
    if missing:
        graph.set_embeddings(dict(zip(missing, vectors[1:])))
    candidates = _route_candidates(
        graph, hierarchy=hierarchy, query_vector=query_vector, route_width=route_width
    )
    scores = {
        memory_id: _cosine(query_vector, graph.units[memory_id].embedding or ())
        for memory_id in candidates
    }
    seeds = tuple(
        sorted(scores, key=lambda memory_id: (-scores[memory_id], memory_id))[:top_k]
    )
    groups: list[tuple[str, ...]] = []
    ordered_memories: list[str] = []
    seen: set[str] = set()
    for seed_id in seeds:
        group = _relation_closure(graph, seed_id, depth=closure_depth)
        ordered_group = tuple(
            sorted(group, key=lambda memory_id: (memory_id != seed_id, memory_id))
        )
        groups.append(ordered_group)
        for memory_id in ordered_group:
            if memory_id not in seen:
                seen.add(memory_id)
                ordered_memories.append(memory_id)
    selected = set(ordered_memories)
    relation_ids = tuple(
        (
            relation.id
            for relation in graph.relations.values()
            if relation.source in selected and relation.target in selected
        )
    )
    source_ids = tuple(
        sorted(
            {
                source_id
                for memory_id in ordered_memories
                for source_id in graph.units[memory_id].source_refs
            },
            key=lambda source_id: graph.sources.get(source_id).sequence,
        )
    )
    return RetrievedSubgraph(
        query=normalized_query,
        seed_ids=seeds,
        memory_ids=tuple(ordered_memories),
        relation_ids=relation_ids,
        source_ids=source_ids,
        closure_groups=tuple(groups),
        scores=MappingProxyType(scores),
    )


def retrieve_memory_branches(
    graph: MemoryGraph,
    *,
    queries: Sequence[str],
    embedding_provider: EmbeddingProvider,
    hierarchy: MemoryHierarchy,
    top_k: int = 8,
    closure_depth: int = 1,
) -> RetrievedSubgraph:
    if top_k <= 0:
        raise ValueError("Retrieval top_k must be positive")
    if closure_depth < 0:
        raise ValueError("Closure depth must be non-negative")
    normalized_queries = tuple((query.strip() for query in queries if query.strip()))
    combined_query = "\n".join(normalized_queries)
    if not normalized_queries or not graph.units:
        return RetrievedSubgraph(
            query=combined_query,
            seed_ids=(),
            memory_ids=(),
            relation_ids=(),
            source_ids=(),
            closure_groups=(),
            scores=MappingProxyType({}),
        )
    missing = [
        memory_id for memory_id, unit in graph.units.items() if unit.embedding is None
    ]
    batch_texts = [
        *normalized_queries,
        *(graph.units[memory_id].content for memory_id in missing),
    ]
    vectors = embedding_provider.embed(batch_texts)
    if len(vectors) != len(batch_texts):
        raise RuntimeError("Embedding provider returned the wrong number of vectors")
    query_vectors = tuple(
        (_normalize_vector(vector) for vector in vectors[: len(normalized_queries)])
    )
    if missing:
        graph.set_embeddings(dict(zip(missing, vectors[len(normalized_queries) :])))
    branch_scores = tuple(
        (
            {
                memory_id: _cosine(query_vector, unit.embedding or ())
                for memory_id, unit in graph.units.items()
            }
            for query_vector in query_vectors
        )
    )
    branches = tuple(
        (
            tuple(sorted(scores, key=lambda memory_id: (-scores[memory_id], memory_id)))
            for scores in branch_scores
        )
    )
    cores = _round_robin_unique(branches, limit=top_k)
    combined_scores = {
        memory_id: max((scores[memory_id] for scores in branch_scores))
        for memory_id in graph.units
    }
    core_set = set(cores)
    expanded: set[str] = set()
    for memory_id in cores:
        expanded.update(_relation_closure(graph, memory_id, depth=closure_depth))
        expanded.update(_deepest_local_cluster_members(hierarchy, memory_id))
    expanded.difference_update(core_set)
    branch_ranks = tuple(
        (
            {memory_id: rank for rank, memory_id in enumerate(branch)}
            for branch in branches
        )
    )
    ordered_expansion = tuple(
        sorted(
            expanded,
            key=lambda memory_id: (
                min((ranks[memory_id] for ranks in branch_ranks)),
                memory_id,
            ),
        )
    )
    memory_ids = (*cores, *ordered_expansion)
    selected = set(memory_ids)
    relation_ids = tuple(
        (
            relation.id
            for relation in graph.relations.values()
            if relation.source in selected and relation.target in selected
        )
    )
    source_ids = tuple(
        sorted(
            {
                source_id
                for memory_id in memory_ids
                for source_id in graph.units[memory_id].source_refs
            },
            key=lambda source_id: graph.sources.get(source_id).sequence,
        )
    )
    return RetrievedSubgraph(
        query=combined_query,
        seed_ids=cores,
        memory_ids=memory_ids,
        relation_ids=relation_ids,
        source_ids=source_ids,
        closure_groups=tuple(((memory_id,) for memory_id in memory_ids)),
        scores=MappingProxyType(combined_scores),
    )


def _route_candidates(
    graph: MemoryGraph,
    *,
    hierarchy: MemoryHierarchy | None,
    query_vector: tuple[float, ...],
    route_width: int,
) -> tuple[str, ...]:
    if hierarchy is None or not hierarchy.root.child_ids:
        return tuple(graph.units)
    cluster_level = list(hierarchy.children_of(hierarchy.root_id))
    while cluster_level and any((cluster.child_ids for cluster in cluster_level)):
        next_level = []
        for cluster in cluster_level:
            children = hierarchy.children_of(cluster.id)
            if children:
                next_level.extend(children)
            else:
                next_level.append(cluster)
        cluster_level = _rank_clusters(graph, next_level, query_vector)[:route_width]
    ranked = _rank_clusters(graph, cluster_level, query_vector)[:route_width]
    routed = {
        memory_id
        for cluster in ranked
        for memory_id in cluster.member_ids
        if memory_id in graph.units
    }
    return tuple(sorted(routed)) if routed else tuple(graph.units)


def _rank_clusters(graph: MemoryGraph, clusters, query_vector: tuple[float, ...]):
    return sorted(
        clusters,
        key=lambda cluster: (
            -_cosine(query_vector, _cluster_centroid(graph, cluster.member_ids)),
            cluster.id,
        ),
    )


def _cluster_centroid(
    graph: MemoryGraph, memory_ids: Sequence[str]
) -> tuple[float, ...]:
    vectors = [
        graph.units[memory_id].embedding
        for memory_id in memory_ids
        if memory_id in graph.units and graph.units[memory_id].embedding is not None
    ]
    if not vectors:
        return ()
    width = len(vectors[0])
    compatible = [
        vector for vector in vectors if vector is not None and len(vector) == width
    ]
    if not compatible:
        return ()
    return tuple(
        (
            sum((vector[index] for vector in compatible)) / len(compatible)
            for index in range(width)
        )
    )


def _relation_closure(graph: MemoryGraph, seed_id: str, *, depth: int) -> set[str]:
    seen = {seed_id}
    wave = {seed_id}
    for _ in range(depth):
        next_wave = {
            neighbor
            for memory_id in wave
            for neighbor in graph.neighbors(memory_id)
            if neighbor not in seen
        }
        if not next_wave:
            break
        seen.update(next_wave)
        wave = next_wave
    return seen


def _round_robin_unique(
    branches: Sequence[Sequence[str]], *, limit: int
) -> tuple[str, ...]:
    queues = tuple((tuple(dict.fromkeys(branch)) for branch in branches if branch))
    cursors = [0] * len(queues)
    selected: list[str] = []
    seen: set[str] = set()
    while len(selected) < limit:
        advanced = False
        for index, queue in enumerate(queues):
            while cursors[index] < len(queue) and queue[cursors[index]] in seen:
                cursors[index] += 1
            if cursors[index] >= len(queue):
                continue
            memory_id = queue[cursors[index]]
            cursors[index] += 1
            selected.append(memory_id)
            seen.add(memory_id)
            advanced = True
            if len(selected) >= limit:
                break
        if not advanced:
            break
    return tuple(selected)


def _deepest_local_cluster_members(
    hierarchy: MemoryHierarchy, memory_id: str
) -> tuple[str, ...]:
    depths = _cluster_depths(hierarchy)
    candidates = tuple(
        (
            cluster
            for cluster in hierarchy.clusters.values()
            if cluster.id != hierarchy.root_id
            and memory_id in cluster.member_ids
            and (len(cluster.member_ids) > 1)
        )
    )
    if not candidates:
        return ()
    deepest = min(
        candidates,
        key=lambda cluster: (-depths[cluster.id], len(cluster.member_ids), cluster.id),
    )
    return deepest.member_ids


def _cluster_depths(hierarchy: MemoryHierarchy) -> Mapping[str, int]:
    depths = {hierarchy.root_id: 0}
    wave = [hierarchy.root_id]
    while wave:
        parent_id = wave.pop(0)
        for child in hierarchy.children_of(parent_id):
            depths[child.id] = depths[parent_id] + 1
            wave.append(child.id)
    return MappingProxyType(depths)


def _embedding_cache_key(model: str, text: str) -> str:
    return f"{model}:{sha1(text.encode('utf-8')).hexdigest()}"


def _normalize_vector(vector: Sequence[float]) -> tuple[float, ...]:
    normalized = tuple((float(value) for value in vector))
    if not normalized or any((not math.isfinite(value) for value in normalized)):
        raise ValueError("Embedding must be a non-empty finite vector")
    return normalized


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum((a * b for a, b in zip(left, right)))
    left_norm = math.sqrt(sum((value * value for value in left)))
    right_norm = math.sqrt(sum((value * value for value in right)))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
