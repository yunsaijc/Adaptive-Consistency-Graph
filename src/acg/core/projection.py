from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from hashlib import sha1
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Protocol, Sequence

from .lineage import RequirementLineageIndex
from .relations import cosine_similarity
from .schema import MemoryUnit
from .source_context import RequirementAnchor
from .store import source_payload_text
from .view import DecisionView

if TYPE_CHECKING:
    from .plugin import AdaptiveConsistencyGraph
PROJECTION_VERSION = "projection.v1"
TEMPORAL_CURRENT = "current_event"
TEMPORAL_LATEST = "latest_in_scope"
TEMPORAL_EARLIER = "earlier"
TEMPORAL_UNSCOPED = "unscoped"


@dataclass(frozen=True)
class SupportIncidence:
    requirement_id: str
    object_scope_id: str | None
    memory_id: str
    event_id: str | None
    temporal_stratum: str
    route_basis: str
    source_refs: tuple[str, ...]
    binding_basis: str = "origin_lineage"
    lineage_source_ids: tuple[str, ...] = ()
    lineage_tool_call_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectionNode:
    id: str
    kind: str
    label: str
    refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectionEdge:
    parent_id: str
    child_id: str
    kind: str
    provenance: str


@dataclass(frozen=True)
class Projection:
    task_source_id: str
    task_text: str
    requirements: tuple[RequirementAnchor, ...]
    nodes: tuple[ProjectionNode, ...]
    edges: tuple[ProjectionEdge, ...]
    incidences: tuple[SupportIncidence, ...]
    current_event_id: str | None
    current_event_memory_ids: tuple[str, ...]
    unassigned_current_memory_ids: tuple[str, ...]
    memory_content: Mapping[str, str]
    memory_source_refs: Mapping[str, tuple[str, ...]]
    event_sequences: Mapping[str, int]
    omitted_memory_ids: tuple[str, ...] = ()
    omission_reasons: Mapping[str, str] | None = None
    unassigned_candidate_memory_ids: tuple[str, ...] = ()
    lineage_qualified_memory_ids: tuple[str, ...] = ()
    lineage_qualified_current_memory_ids: tuple[str, ...] = ()
    lineage_telemetry: Mapping[str, Mapping[str, Any]] | None = None
    hierarchy_provider: str = "event_stage"
    projection_version: str = PROJECTION_VERSION

    @property
    def shared_candidate_memory_ids(self) -> tuple[str, ...]:
        return self.unassigned_candidate_memory_ids

    def incidences_for_requirement(
        self, requirement_id: str
    ) -> tuple[SupportIncidence, ...]:
        return tuple(
            (
                incidence
                for incidence in self.incidences
                if incidence.requirement_id == requirement_id
            )
        )

    def memory_ids(self) -> tuple[str, ...]:
        return tuple(self.memory_content)


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        raise NotImplementedError


def build_projection(
    plugin: AdaptiveConsistencyGraph,
    view: DecisionView,
    *,
    requirements: Sequence[RequirementAnchor],
    current_event_id: str | None,
    max_requirement_candidates: int = 8,
    excluded_source_ids: Iterable[str] = (),
    requirement_vectors: Sequence[Sequence[float]] | None = None,
    lineage_index: RequirementLineageIndex | None = None,
) -> Projection:
    if max_requirement_candidates <= 0:
        raise ValueError("max_requirement_candidates must be positive")
    excluded = set(excluded_source_ids)
    lineage = lineage_index or getattr(plugin, "lineage", None)
    if lineage is not None:
        lineage.set_requirements(requirements)
    current_ids = tuple(
        (
            memory_id
            for memory_id in _current_event_memory_ids(plugin, current_event_id)
            if _eligible_memory(plugin.graph.units.get(memory_id), set())
        )
    )
    view_ids = _ordered_unique(
        (*view.retrieved_memory_ids, *(memory.id for memory in view.memories))
    )
    requirement_seed_ids = tuple(
        (
            memory_id
            for memory_id in view_ids
            if _eligible_memory(plugin.graph.units.get(memory_id), excluded)
        )
    )
    current_event = (
        plugin.organization.events.get(current_event_id)
        if current_event_id is not None
        else None
    )
    lineage_qualified_current_ids = tuple(
        (
            memory_id
            for memory_id in current_ids
            if lineage is not None
            and current_event is not None
            and any(
                (
                    origin.origin_requirement_ids
                    for origin in lineage.origins_for_memory_event(
                        plugin.graph.units[memory_id], current_event
                    )
                )
            )
        )
    )
    lineage_qualified_view_ids = tuple(
        (
            memory_id
            for memory_id in requirement_seed_ids
            if _memory_has_requirement_lineage(
                plugin, plugin.graph.units[memory_id], lineage=lineage
            )
        )
    )
    candidate_universe_ids = _ordered_unique((*requirement_seed_ids, *current_ids))
    requirement_candidate_ids = _add_identity_and_event_closure(
        plugin,
        (*lineage_qualified_view_ids, *lineage_qualified_current_ids),
        current_event_id=None,
        excluded_source_ids=excluded,
        candidate_universe_ids=candidate_universe_ids,
        excluded_memory_ids=set(current_ids).difference(
            (*lineage_qualified_view_ids, *lineage_qualified_current_ids)
        ),
    )
    requirement_candidate_ids = tuple(
        (
            memory_id
            for memory_id in requirement_candidate_ids
            if _memory_has_requirement_lineage(
                plugin, plugin.graph.units[memory_id], lineage=lineage
            )
        )
    )
    qualified_ids_by_requirement = {
        requirement.id: tuple(
            (
                memory_id
                for memory_id in requirement_candidate_ids
                if requirement.id
                in _memory_requirement_ids(
                    plugin, plugin.graph.units[memory_id], lineage=lineage
                )
            )
        )
        for requirement in requirements
    }
    expanded_ids = _ordered_unique((*current_ids, *requirement_candidate_ids))
    vectors = tuple(requirement_vectors or ())
    if requirements and (not vectors):
        vectors = tuple(
            (
                tuple(vector)
                for vector in plugin.embedding_provider.embed(
                    tuple(
                        (
                            _requirement_query(requirement)
                            for requirement in requirements
                        )
                    )
                )
            )
        )
    if len(vectors) != len(requirements):
        raise ValueError("Requirement vector count does not match requirements")
    identity_memberships = {
        memory_id: _memory_identity_keys(plugin.graph.units[memory_id])
        for memory_id in expanded_ids
    }
    object_ids = {
        key: _stable_object_id(key)
        for key in sorted(
            {key for keys in identity_memberships.values() for key in keys}
        )
    }
    memory_events = _memory_events(plugin, expanded_ids)
    event_sequences = {
        event_id: plugin.organization.events[event_id].sequence
        for event_id in {
            event_id
            for event_ids in memory_events.values()
            for event_id in event_ids
            if event_id in plugin.organization.events
        }
    }
    selected_by_requirement: dict[str, tuple[str, ...]] = {}
    for requirement, vector in zip(requirements, vectors):
        candidate_ids = qualified_ids_by_requirement[requirement.id]
        history_ids = tuple(
            (memory_id for memory_id in candidate_ids if memory_id not in current_ids)
        )
        ranked_history = _rank_requirement_candidates(
            plugin, requirement, vector, history_ids, limit=max_requirement_candidates
        )
        selected_by_requirement[requirement.id] = _ordered_unique(
            (
                *ranked_history,
                *(memory_id for memory_id in candidate_ids if memory_id in current_ids),
            )
        )
    latest_event_by_object: dict[tuple[str, str], int] = {}
    for memory_id, keys in identity_memberships.items():
        for key in keys:
            event_values = tuple(
                (
                    event_sequences.get(event_id, -1)
                    for event_id in memory_events.get(memory_id, ())
                )
            )
            latest_event_by_object[key] = max(
                (latest_event_by_object.get(key, -1), *event_values), default=-1
            )
    incidences: list[SupportIncidence] = []
    unassigned_candidates: list[str] = []
    ranked_ids = _ordered_unique(
        (
            memory_id
            for requirement in requirements
            for memory_id in selected_by_requirement[requirement.id]
        )
    )
    requirement_by_id = {item.id: item for item in requirements}
    for memory_id in ranked_ids:
        memory = plugin.graph.units[memory_id]
        event_ids: tuple[str | None, ...] = tuple(
            sorted(
                memory_events.get(memory_id, ()),
                key=lambda event_id: (event_sequences.get(event_id, -1), event_id),
            )
        ) or (None,)
        memory_had_origin = False
        for event_id in event_ids:
            origins = (
                lineage.origins_for_memory_event(
                    memory, plugin.organization.events.get(event_id)
                )
                if lineage is not None
                else ()
            )
            if not origins:
                continue
            memory_had_origin = True
            keys = identity_memberships.get(memory_id, ())
            scopes = tuple((object_ids[key] for key in keys)) or (None,)
            for origin in origins:
                for requirement_id in origin.origin_requirement_ids:
                    if requirement_id not in requirement_by_id:
                        continue
                    for object_scope_id in scopes:
                        object_key = _object_key_for_id(object_ids, object_scope_id)
                        stratum = _temporal_stratum(
                            event_id,
                            current_event_id=current_event_id,
                            object_key=object_key,
                            latest_event_by_object=latest_event_by_object,
                            event_sequences=event_sequences,
                        )
                        incidences.append(
                            SupportIncidence(
                                requirement_id=requirement_id,
                                object_scope_id=object_scope_id,
                                memory_id=memory_id,
                                event_id=event_id,
                                temporal_stratum=stratum,
                                route_basis=_route_basis(
                                    memory_id,
                                    current_ids=current_ids,
                                    base_ids=requirement_candidate_ids,
                                ),
                                source_refs=tuple(memory.source_refs),
                                binding_basis="origin_lineage",
                                lineage_source_ids=(origin.source_id,),
                                lineage_tool_call_ids=(origin.tool_call_id,)
                                if origin.tool_call_id is not None
                                else (),
                            )
                        )
        if not memory_had_origin and memory_id not in current_ids:
            unassigned_candidates.append(memory_id)
    incidences = _dedupe_incidences(incidences)
    linked_ids = {incidence.memory_id for incidence in incidences}
    unassigned_current = tuple(
        (
            memory_id
            for memory_id in current_ids
            if memory_id in expanded_ids and memory_id not in linked_ids
        )
    )
    nodes, edges = _build_nodes_and_edges(
        plugin,
        requirements=requirements,
        incidences=incidences,
        identity_memberships=identity_memberships,
        object_ids=object_ids,
        memory_events=memory_events,
        current_event_id=current_event_id,
        event_sequences=event_sequences,
        current_event_memory_ids=current_ids,
    )
    return Projection(
        task_source_id=plugin.task.id,
        task_text=source_payload_text(plugin.task),
        requirements=tuple(requirements),
        nodes=nodes,
        edges=edges,
        incidences=tuple(incidences),
        current_event_id=current_event_id,
        current_event_memory_ids=current_ids,
        unassigned_current_memory_ids=unassigned_current,
        memory_content={
            memory_id: plugin.graph.units[memory_id].content
            for memory_id in expanded_ids
        },
        memory_source_refs={
            memory_id: tuple(plugin.graph.units[memory_id].source_refs)
            for memory_id in expanded_ids
        },
        event_sequences=event_sequences,
        omission_reasons={},
        unassigned_candidate_memory_ids=tuple(unassigned_candidates),
        lineage_qualified_memory_ids=_ordered_unique(
            (*lineage_qualified_view_ids, *lineage_qualified_current_ids)
        ),
        lineage_qualified_current_memory_ids=lineage_qualified_current_ids,
        lineage_telemetry=lineage.telemetry if lineage is not None else {},
        projection_version=PROJECTION_VERSION,
    )


def _rank_requirement_candidates(
    plugin: AdaptiveConsistencyGraph,
    requirement: RequirementAnchor,
    query_vector: Sequence[float],
    memory_ids: Sequence[str],
    *,
    limit: int,
) -> tuple[str, ...]:
    scored: list[tuple[float, int, str]] = []
    for memory_id in memory_ids:
        memory = plugin.graph.units[memory_id]
        semantic = cosine_similarity(query_vector, memory.embedding or ())
        sequence = max(
            (
                plugin.organization.events[event_id].sequence
                for event_id in _memory_events(plugin, (memory_id,)).get(memory_id, ())
                if event_id in plugin.organization.events
            ),
            default=-1,
        )
        scored.append((semantic, sequence, memory_id))
    del requirement
    return tuple(
        (
            memory_id
            for _, _, memory_id in sorted(
                scored, key=lambda item: (-item[0], -item[1], item[2])
            )[:limit]
        )
    )


def _build_nodes_and_edges(
    plugin: AdaptiveConsistencyGraph,
    *,
    requirements: Sequence[RequirementAnchor],
    incidences: Sequence[SupportIncidence],
    identity_memberships: Mapping[str, tuple[tuple[str, str], ...]],
    object_ids: Mapping[tuple[str, str], str],
    memory_events: Mapping[str, tuple[str, ...]],
    current_event_id: str | None,
    event_sequences: Mapping[str, int],
    current_event_memory_ids: Sequence[str],
) -> tuple[tuple[ProjectionNode, ...], tuple[ProjectionEdge, ...]]:
    nodes: dict[str, ProjectionNode] = {}
    edges: dict[tuple[str, str, str], ProjectionEdge] = {}

    def add_node(node: ProjectionNode) -> None:
        nodes.setdefault(node.id, node)

    def add_edge(parent: str, child: str, kind: str, provenance: str) -> None:
        key = (parent, child, kind)
        edges.setdefault(key, ProjectionEdge(parent, child, kind, provenance))

    task_id = f"task:{plugin.task.id}"
    add_node(
        ProjectionNode(
            task_id, "task", source_payload_text(plugin.task), (plugin.task.id,)
        )
    )
    for requirement in requirements:
        requirement_id = f"requirement:{requirement.id}"
        add_node(
            ProjectionNode(
                requirement_id,
                "requirement",
                requirement.exact_text,
                (requirement.source_id,),
            )
        )
        add_edge(task_id, requirement_id, "contains", "exact_task_span")
    for incidence in incidences:
        requirement_id = f"requirement:{incidence.requirement_id}"
        evidence_id = f"evidence:{incidence.memory_id}"
        support_id = _support_node_id(incidence)
        time_id = _time_node_id(incidence)
        memory = plugin.graph.units[incidence.memory_id]
        add_node(
            ProjectionNode(
                evidence_id, "evidence", memory.content, tuple(memory.source_refs)
            )
        )
        add_node(
            ProjectionNode(
                support_id,
                "support",
                incidence.route_basis,
                tuple(incidence.source_refs),
            )
        )
        add_node(
            ProjectionNode(
                time_id,
                "time",
                incidence.temporal_stratum,
                (incidence.event_id,) if incidence.event_id is not None else (),
            )
        )
        add_edge(requirement_id, support_id, "has_support", incidence.route_basis)
        add_edge(support_id, evidence_id, "support_evidence", incidence.route_basis)
        add_edge(support_id, time_id, "temporal_scope", "source_event_membership")
        add_edge(requirement_id, evidence_id, "supports", incidence.route_basis)
        if incidence.object_scope_id is not None:
            object_id = f"object:{incidence.object_scope_id}"
            add_node(
                ProjectionNode(
                    object_id,
                    "object",
                    _object_label_for_id(object_ids, incidence.object_scope_id),
                    (incidence.object_scope_id,),
                )
            )
            add_edge(support_id, object_id, "support_scope", incidence.route_basis)
            add_edge(requirement_id, object_id, "support_scope", incidence.route_basis)
            add_edge(object_id, evidence_id, "object_evidence", incidence.route_basis)
        if incidence.event_id is not None:
            event_id = f"event:{incidence.event_id}"
            event = plugin.organization.events[incidence.event_id]
            add_node(
                ProjectionNode(
                    event_id, "event", event.content, tuple(event.source_ids)
                )
            )
            add_edge(time_id, event_id, "event_scope", "event_provenance")
            add_edge(event_id, evidence_id, "event_evidence", "event_provenance")
        for source_id in memory.source_refs:
            source_node_id = f"source:{source_id}"
            source = plugin.sources.get(source_id)
            add_node(
                ProjectionNode(
                    source_node_id, "source", source_payload_text(source), (source_id,)
                )
            )
            add_edge(evidence_id, source_node_id, "source_ref", "memory_source_refs")
    for memory_id in current_event_memory_ids:
        memory = plugin.graph.units[memory_id]
        evidence_id = f"evidence:{memory_id}"
        add_node(
            ProjectionNode(
                evidence_id, "evidence", memory.content, tuple(memory.source_refs)
            )
        )
        current_scopes = tuple(
            (object_ids[key] for key in identity_memberships.get(memory_id, ()))
        ) or (None,)
        for object_scope_id in current_scopes:
            support_id = _visibility_support_node_id(
                memory_id, object_scope_id=object_scope_id, event_id=current_event_id
            )
            time_id = f"time:{support_id.split(':', 1)[1]}"
            add_node(
                ProjectionNode(
                    support_id,
                    "support",
                    "current_event_visibility",
                    tuple(memory.source_refs),
                )
            )
            add_node(
                ProjectionNode(
                    time_id,
                    "time",
                    TEMPORAL_CURRENT,
                    (current_event_id,) if current_event_id is not None else (),
                )
            )
            add_edge(
                task_id,
                support_id,
                "current_event_visibility",
                "current_event_visibility",
            )
            add_edge(
                support_id, evidence_id, "support_evidence", "current_event_visibility"
            )
            add_edge(support_id, time_id, "temporal_scope", "current_event_visibility")
            if object_scope_id is not None:
                object_id = f"object:{object_scope_id}"
                add_node(
                    ProjectionNode(
                        object_id,
                        "object",
                        _object_label_for_id(object_ids, object_scope_id),
                        (object_scope_id,),
                    )
                )
                add_edge(
                    support_id, object_id, "support_scope", "current_event_visibility"
                )
                add_edge(
                    object_id,
                    evidence_id,
                    "object_evidence",
                    "current_event_visibility",
                )
            if current_event_id is not None:
                add_edge(
                    time_id,
                    f"event:{current_event_id}",
                    "event_scope",
                    "current_event_visibility",
                )
        if (
            current_event_id is not None
            and current_event_id in plugin.organization.events
        ):
            event = plugin.organization.events[current_event_id]
            event_node_id = f"event:{current_event_id}"
            add_node(
                ProjectionNode(
                    event_node_id, "event", event.content, tuple(event.source_ids)
                )
            )
            add_edge(
                event_node_id, evidence_id, "event_evidence", "current_event_visibility"
            )
        for source_id in memory.source_refs:
            source_node_id = f"source:{source_id}"
            source = plugin.sources.get(source_id)
            add_node(
                ProjectionNode(
                    source_node_id, "source", source_payload_text(source), (source_id,)
                )
            )
            add_edge(evidence_id, source_node_id, "source_ref", "memory_source_refs")
    del identity_memberships, memory_events, current_event_id, event_sequences
    ordered_nodes = tuple(sorted(nodes.values(), key=lambda node: (node.kind, node.id)))
    ordered_edges = tuple(
        sorted(
            edges.values(), key=lambda edge: (edge.parent_id, edge.child_id, edge.kind)
        )
    )
    return (ordered_nodes, ordered_edges)


def _add_identity_and_event_closure(
    plugin: AdaptiveConsistencyGraph,
    base_ids: Sequence[str],
    *,
    current_event_id: str | None,
    excluded_source_ids: set[str],
    excluded_memory_ids: set[str] | None = None,
    candidate_universe_ids: Sequence[str] | None = None,
) -> tuple[str, ...]:
    excluded_memory_ids = excluded_memory_ids or set()
    identities = {
        key
        for memory_id in base_ids
        for key in _memory_identity_keys(plugin.graph.units[memory_id])
    }
    event_ids = {
        event_id
        for memory_id in base_ids
        for event_id in _memory_events(plugin, (memory_id,)).get(memory_id, ())
    }
    ordered = list(base_ids)
    universe = (
        tuple(candidate_universe_ids)
        if candidate_universe_ids is not None
        else tuple(plugin.graph.units)
    )
    for memory_id in universe:
        memory = plugin.graph.units.get(memory_id)
        if memory is None:
            continue
        if memory_id in excluded_memory_ids and memory_id not in base_ids:
            continue
        if memory_id not in base_ids and (
            not _eligible_memory(memory, excluded_source_ids)
        ):
            continue
        memory_keys = set(_memory_identity_keys(memory))
        memory_event_ids = set(_memory_events(plugin, (memory_id,)).get(memory_id, ()))
        if identities.intersection(memory_keys) or event_ids.intersection(
            memory_event_ids
        ):
            if memory_id not in ordered:
                ordered.append(memory_id)
    if current_event_id is not None:
        for memory_id in _current_event_memory_ids(plugin, current_event_id):
            if (
                memory_id not in ordered
                and memory_id in universe
                and _eligible_memory(plugin.graph.units.get(memory_id), set())
            ):
                ordered.append(memory_id)
    return tuple(ordered)


def _current_event_memory_ids(
    plugin: AdaptiveConsistencyGraph, current_event_id: str | None
) -> tuple[str, ...]:
    if current_event_id is None:
        return ()
    event = plugin.organization.events.get(current_event_id)
    return tuple(event.evidence_ids) if event is not None else ()


def _memory_events(
    plugin: AdaptiveConsistencyGraph, memory_ids: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    wanted = set(memory_ids)
    result: dict[str, list[str]] = {memory_id: [] for memory_id in memory_ids}
    for event in plugin.organization.events.values():
        for memory_id in event.evidence_ids:
            if memory_id in wanted:
                result[memory_id].append(event.id)
    return {memory_id: tuple(event_ids) for memory_id, event_ids in result.items()}


def _memory_has_requirement_lineage(
    plugin: AdaptiveConsistencyGraph,
    memory: MemoryUnit,
    *,
    lineage: RequirementLineageIndex | None,
) -> bool:
    if lineage is None:
        return False
    event_ids = _memory_events(plugin, (memory.id,)).get(memory.id, ())
    if event_ids:
        return any(
            (
                origin.origin_requirement_ids
                for event_id in event_ids
                if event_id in plugin.organization.events
                for origin in lineage.origins_for_memory_event(
                    memory, plugin.organization.events[event_id]
                )
            )
        )
    return any(
        (
            origin.origin_requirement_ids
            for origin in lineage.origins_for_memory_event(memory, None)
        )
    )


def _memory_requirement_ids(
    plugin: AdaptiveConsistencyGraph,
    memory: MemoryUnit,
    *,
    lineage: RequirementLineageIndex | None,
) -> frozenset[str]:
    if lineage is None:
        return frozenset()
    event_ids = _memory_events(plugin, (memory.id,)).get(memory.id, ())
    records = (
        origin
        for event_id in event_ids
        if event_id in plugin.organization.events
        for origin in lineage.origins_for_memory_event(
            memory, plugin.organization.events[event_id]
        )
    )
    if not event_ids:
        records = lineage.origins_for_memory_event(memory, None)
    return frozenset(
        (
            requirement_id
            for origin in records
            for requirement_id in origin.origin_requirement_ids
        )
    )


def _temporal_stratum(
    event_id: str | None,
    *,
    current_event_id: str | None,
    object_key: tuple[str, str] | None,
    latest_event_by_object: Mapping[tuple[str, str], int],
    event_sequences: Mapping[str, int],
) -> str:
    if event_id is None:
        return TEMPORAL_UNSCOPED
    if event_id == current_event_id:
        return TEMPORAL_CURRENT
    if object_key is not None and event_sequences.get(
        event_id, -1
    ) == latest_event_by_object.get(object_key, -2):
        return TEMPORAL_LATEST
    return TEMPORAL_EARLIER


def _route_basis(
    memory_id: str, *, current_ids: Sequence[str], base_ids: Sequence[str]
) -> str:
    if memory_id in base_ids:
        return "requirement_route"
    if memory_id in current_ids:
        return "current_event"
    return "identity_closure"


def _eligible_memory(memory: MemoryUnit | None, excluded_source_ids: set[str]) -> bool:
    return (
        memory is not None
        and (not excluded_source_ids.intersection(memory.source_refs))
        and (not _projection_wrapper(memory))
    )


def _projection_wrapper(memory: MemoryUnit) -> bool:
    return any(
        (
            path == "/event_type" and value == "tool_observation_batch"
            for path, value in _structured_fields(memory.content)
        )
    )


def _memory_identity_keys(memory: MemoryUnit) -> tuple[tuple[str, str], ...]:
    keys: list[tuple[str, str]] = []
    for path, value in _structured_fields(memory.content):
        terminal = path.rsplit("/", 1)[-1].replace("~1", "/").replace("~0", "~").lower()
        normalized = _identity_value(value)
        if (
            not normalized
            or not _is_identity_terminal(terminal)
            or "/function/" in path.lower()
        ):
            continue
        key = (terminal, normalized)
        if key not in keys:
            keys.append(key)
    return tuple(keys)


def _is_identity_terminal(terminal: str) -> bool:
    if terminal in {
        "id",
        "uuid",
        "key",
        "call_id",
        "request_id",
        "trace_id",
        "tool_call_id",
    }:
        return False
    if set(terminal.split("_")) & {
        "account",
        "actor",
        "agent",
        "call",
        "conversation",
        "customer",
        "message",
        "request",
        "session",
        "tenant",
        "tool",
        "trace",
        "user",
    }:
        return False
    return (
        terminal == "docid"
        or bool(re.search("(?:^|_)(?:id|uuid|key)$", terminal))
        or terminal.endswith("_path")
    )


def _structured_fields(content: str) -> tuple[tuple[str, Any], ...]:
    fields: list[tuple[str, Any]] = []
    for line in str(content).splitlines():
        match = re.fullmatch("(/\\S*)\\s*=\\s*(.+)", line.strip())
        if match is None:
            continue
        try:
            fields.append((match.group(1), json.loads(match.group(2))))
        except json.JSONDecodeError:
            continue
    return tuple(fields)


def _identity_value(value: Any) -> str:
    if value is None or isinstance(value, (dict, list)):
        return ""
    return str(value).strip()


def _stable_object_id(key: tuple[str, str]) -> str:
    digest = sha1(f"{key[0]}\x00{key[1]}".encode("utf-8")).hexdigest()[:12]
    return f"object-{digest}"


def _object_key_for_id(
    object_ids: Mapping[tuple[str, str], str], object_scope_id: str | None
) -> tuple[str, str] | None:
    if object_scope_id is None:
        return None
    return next(
        (key for key, value in object_ids.items() if value == object_scope_id), None
    )


def _object_label_for_id(
    object_ids: Mapping[tuple[str, str], str], object_scope_id: str
) -> str:
    key = _object_key_for_id(object_ids, object_scope_id)
    if key is None:
        return object_scope_id
    terminal, value = key
    return f"{terminal}={value}"


def _requirement_query(requirement: RequirementAnchor) -> str:
    return requirement.exact_text.strip()


def _dedupe_incidences(
    incidences: Sequence[SupportIncidence],
) -> list[SupportIncidence]:
    result: dict[tuple[str, str | None, str, str | None, str], SupportIncidence] = {}
    for incidence in incidences:
        key = (
            incidence.requirement_id,
            incidence.object_scope_id,
            incidence.memory_id,
            incidence.event_id,
            incidence.temporal_stratum,
        )
        existing = result.get(key)
        if existing is None:
            result[key] = incidence
            continue
        result[key] = replace(
            existing,
            lineage_source_ids=tuple(
                dict.fromkeys(
                    (*existing.lineage_source_ids, *incidence.lineage_source_ids)
                )
            ),
            lineage_tool_call_ids=tuple(
                dict.fromkeys(
                    (*existing.lineage_tool_call_ids, *incidence.lineage_tool_call_ids)
                )
            ),
        )
    return list(result.values())


def _support_node_id(incidence: SupportIncidence) -> str:
    return _branch_node_id(
        incidence.requirement_id,
        incidence.object_scope_id,
        incidence.memory_id,
        incidence.event_id,
        incidence.temporal_stratum,
        incidence.route_basis,
        prefix="support",
    )


def _time_node_id(incidence: SupportIncidence) -> str:
    return _branch_node_id(
        incidence.requirement_id,
        incidence.object_scope_id,
        incidence.memory_id,
        incidence.event_id,
        incidence.temporal_stratum,
        incidence.route_basis,
        prefix="time",
    )


def _visibility_support_node_id(
    memory_id: str, *, object_scope_id: str | None, event_id: str | None
) -> str:
    return _branch_node_id(
        None,
        object_scope_id,
        memory_id,
        event_id,
        TEMPORAL_CURRENT,
        "current_event_visibility",
        prefix="support",
    )


def _branch_node_id(
    requirement_id: str | None,
    object_scope_id: str | None,
    memory_id: str,
    event_id: str | None,
    temporal_stratum: str,
    route_basis: str,
    *,
    prefix: str,
) -> str:
    payload = "\x00".join(
        (
            requirement_id or "",
            object_scope_id or "",
            memory_id,
            event_id or "",
            temporal_stratum,
            route_basis,
        )
    )
    return f"{prefix}:{sha1(payload.encode('utf-8')).hexdigest()[:16]}"


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


__all__ = [
    "Projection",
    "ProjectionEdge",
    "ProjectionNode",
    "SupportIncidence",
    "TEMPORAL_CURRENT",
    "TEMPORAL_EARLIER",
    "TEMPORAL_LATEST",
    "TEMPORAL_UNSCOPED",
    "build_projection",
]
