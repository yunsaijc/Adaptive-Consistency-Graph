from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from .hierarchy import HierarchyBuilder, StructuralEntropyHierarchyBuilder
from .relations import cosine_similarity
from .schema import MemoryHierarchy, SourceKind
from .store import MemoryGraph, SourceStore


@dataclass(frozen=True)
class EventNode:
    id: str
    sequence: int
    source_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    content: str
    embedding: tuple[float, ...]


@dataclass
class ConceptAnchor:
    id: str
    anchor_texts: tuple[str, ...]
    source_event_id: str
    event_ids: set[str] = field(default_factory=set)
    evidence_ids_by_event: dict[str, set[str]] = field(default_factory=dict)
    embedding: tuple[float, ...] | None = None

    @property
    def text(self) -> str:
        return "\n".join(self.anchor_texts)

    @property
    def materialized(self) -> bool:
        return len(self.event_ids) >= 2


class EventOrganization:
    def __init__(self) -> None:
        self._events: dict[str, EventNode] = {}
        self._concepts: dict[str, ConceptAnchor] = {}
        self._concept_by_anchor: dict[tuple[str, ...], str] = {}
        self._active_concepts_by_event: dict[str, set[str]] = {}
        self._next_event_number = 1
        self._next_concept_number = 1

    @property
    def events(self) -> Mapping[str, EventNode]:
        return MappingProxyType(self._events)

    @property
    def concepts(self) -> Mapping[str, ConceptAnchor]:
        return MappingProxyType(self._concepts)

    @property
    def materialized_concept_ids(self) -> frozenset[str]:
        return frozenset(
            (concept.id for concept in self._concepts.values() if concept.materialized)
        )

    @property
    def ownership_edges(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (
                (event.id, evidence_id)
                for event in self._events.values()
                for evidence_id in event.evidence_ids
            )
        )

    @property
    def sequence_edges(self) -> tuple[tuple[str, str], ...]:
        ordered = tuple(self._events)
        return tuple(zip(ordered, ordered[1:]))

    @property
    def navigation_edges(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (
                (concept.id, event_id)
                for concept in self._concepts.values()
                if concept.materialized
                for event_id in sorted(
                    concept.event_ids, key=lambda item: self._events[item].sequence
                )
            )
        )

    def add_event(
        self,
        *,
        source_ids: Iterable[str],
        evidence_ids: Iterable[str],
        content: str,
        embedding: Iterable[float],
    ) -> EventNode:
        event_id = f"event-{self._next_event_number:06d}"
        self._next_event_number += 1
        event = EventNode(
            id=event_id,
            sequence=len(self._events) + 1,
            source_ids=_unique(source_ids),
            evidence_ids=_unique(evidence_ids),
            content=str(content),
            embedding=tuple((float(value) for value in embedding)),
        )
        if not event.source_ids:
            raise ValueError("An event must own at least one source")
        if not event.embedding:
            raise ValueError("An event must have a non-empty embedding")
        self._events[event.id] = event
        self._active_concepts_by_event[event.id] = set()
        return event

    def restore_event(
        self, record: Mapping[str, object], *, embedding: Iterable[float]
    ) -> EventNode:
        event_id = str(record.get("id") or "").strip()
        if not event_id:
            raise ValueError("Restored events require an id")
        sequence = record.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence <= 0:
            raise ValueError("Restored event sequence must be a positive integer")
        event = EventNode(
            id=event_id,
            sequence=sequence,
            source_ids=_unique(record.get("source_ids") or ()),
            evidence_ids=_unique(record.get("evidence_ids") or ()),
            content=str(record.get("content") or ""),
            embedding=tuple((float(value) for value in embedding)),
        )
        if not event.source_ids:
            raise ValueError("Restored events must own at least one source")
        if not event.embedding:
            raise ValueError("Restored events require a non-empty embedding")
        existing = self._events.get(event_id)
        if existing is not None:
            if existing != event:
                raise ValueError(
                    f"Restored event id conflicts with existing event: {event_id}"
                )
            return existing
        self._events[event_id] = event
        self._active_concepts_by_event[event_id] = set()
        suffix = event_id.removeprefix("event-")
        if suffix.isdigit():
            self._next_event_number = max(self._next_event_number, int(suffix) + 1)
        return event

    def create_or_reuse_concept(
        self,
        anchor_texts: Sequence[str],
        *,
        event_id: str,
        embedding: Iterable[float] | None = None,
        active: bool = True,
        evidence_ids: Iterable[str] = (),
    ) -> str:
        self._require_event(event_id)
        key = _unique(anchor_texts)
        if not key:
            raise ValueError("A concept anchor must contain source text")
        concept_id = self._concept_by_anchor.get(key)
        if concept_id is None:
            concept_id = f"concept-{self._next_concept_number:06d}"
            self._next_concept_number += 1
            self._concepts[concept_id] = ConceptAnchor(
                id=concept_id,
                anchor_texts=key,
                source_event_id=event_id,
                event_ids={event_id},
                evidence_ids_by_event={event_id: set(_unique(evidence_ids))},
                embedding=tuple((float(value) for value in embedding))
                if embedding is not None
                else None,
            )
            self._concept_by_anchor[key] = concept_id
        else:
            self._concepts[concept_id].event_ids.add(event_id)
            self._concepts[concept_id].evidence_ids_by_event.setdefault(
                event_id, set()
            ).update(_unique(evidence_ids))
        if active:
            self._active_concepts_by_event[event_id].add(concept_id)
        return concept_id

    def restore_concept(
        self,
        record: Mapping[str, object],
        *,
        embedding: Iterable[float] | None = None,
        active_event_ids: Iterable[str] = (),
    ) -> ConceptAnchor:
        concept_id = str(record.get("id") or "").strip()
        anchor_texts = _unique(record.get("anchor_texts") or ())
        source_event_id = str(record.get("source_event_id") or "").strip()
        if not concept_id or not anchor_texts or (not source_event_id):
            raise ValueError("Restored concepts require id, anchors, and source event")
        event_ids = set(_unique(record.get("event_ids") or ()))
        evidence_by_event = {
            str(event_id): set(_unique(values or ()))
            for event_id, values in (record.get("evidence_ids_by_event") or {}).items()
        }
        if not event_ids:
            event_ids.add(source_event_id)
        if any((event_id not in self._events for event_id in event_ids)):
            raise ValueError(f"Restored concept references unknown event: {concept_id}")
        concept_embedding = (
            tuple((float(value) for value in embedding))
            if embedding is not None
            else None
        )
        concept = ConceptAnchor(
            id=concept_id,
            anchor_texts=anchor_texts,
            source_event_id=source_event_id,
            event_ids=event_ids,
            evidence_ids_by_event=evidence_by_event,
            embedding=concept_embedding,
        )
        existing = self._concepts.get(concept_id)
        if existing is not None and existing != concept:
            raise ValueError(
                f"Restored concept id conflicts with existing concept: {concept_id}"
            )
        self._concepts[concept_id] = concept
        self._concept_by_anchor[anchor_texts] = concept_id
        suffix = concept_id.removeprefix("concept-")
        if suffix.isdigit():
            self._next_concept_number = max(self._next_concept_number, int(suffix) + 1)
        active = set(active_event_ids) or event_ids
        for event_id in active:
            if event_id in self._active_concepts_by_event:
                self._active_concepts_by_event[event_id].add(concept_id)
        return concept

    def attach_concept(
        self,
        concept_id: str,
        *,
        event_id: str,
        active: bool = True,
        evidence_ids: Iterable[str] = (),
    ) -> None:
        self._require_event(event_id)
        try:
            concept = self._concepts[concept_id]
        except KeyError as error:
            raise ValueError(f"Unknown concept: {concept_id}") from error
        concept.event_ids.add(event_id)
        concept.evidence_ids_by_event.setdefault(event_id, set()).update(
            _unique(evidence_ids)
        )
        if active:
            self._active_concepts_by_event[event_id].add(concept_id)

    def active_concept_ids(self, event_id: str) -> tuple[str, ...]:
        self._require_event(event_id)
        return tuple(sorted(self._active_concepts_by_event[event_id]))

    def event_for_source(self, source_id: str) -> EventNode | None:
        return next(
            (
                event
                for event in reversed(tuple(self._events.values()))
                if source_id in event.source_ids
            ),
            None,
        )

    def evidence_ids_for_events(self, event_ids: Sequence[str]) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    evidence_id
                    for event_id in event_ids
                    for evidence_id in self._require_event(event_id).evidence_ids
                )
            )
        )

    def prior_events_for(self, event_id: str) -> tuple[str, ...]:
        current = self._require_event(event_id)
        prior = {
            prior_event_id
            for concept_id in self._active_concepts_by_event[event_id]
            if self._concepts[concept_id].materialized
            for prior_event_id in self._concepts[concept_id].event_ids
            if self._events[prior_event_id].sequence < current.sequence
        }
        return tuple(
            sorted(prior, key=lambda item: self._events[item].sequence, reverse=True)
        )

    def earlier_evidence_ids(self, event_id: str) -> frozenset[str]:
        current = self._require_event(event_id)
        earlier: set[str] = set()
        for concept in self._concepts.values():
            occurrences = sorted(
                (
                    self._events[concept_event_id]
                    for concept_event_id in concept.event_ids
                    if self._events[concept_event_id].sequence <= current.sequence
                ),
                key=lambda item: item.sequence,
            )
            if len(occurrences) < 2:
                continue
            latest_sequence = occurrences[-1].sequence
            for occurrence in occurrences:
                if occurrence.sequence < latest_sequence:
                    earlier.update(concept.evidence_ids_by_event.get(occurrence.id, ()))
        return frozenset(earlier)

    def set_concept_embeddings(self, embeddings: Mapping[str, Iterable[float]]) -> None:
        unknown = set(embeddings) - set(self._concepts)
        if unknown:
            raise ValueError(f"Unknown concepts: {sorted(unknown)}")
        for concept_id, embedding in embeddings.items():
            vector = tuple((float(value) for value in embedding))
            if not vector:
                raise ValueError("Concept embedding must not be empty")
            self._concepts[concept_id].embedding = vector

    def _require_event(self, event_id: str) -> EventNode:
        try:
            return self._events[event_id]
        except KeyError as error:
            raise ValueError(f"Unknown event: {event_id}") from error


def build_event_hierarchy(
    organization: EventOrganization,
    *,
    hierarchy_builder: HierarchyBuilder | None = None,
) -> MemoryHierarchy:
    sources = SourceStore()
    graph = MemoryGraph(sources)
    for event in organization.events.values():
        source_id = f"{event.id}-source"
        sources.append(
            SourceKind.OBSERVATION,
            {"source_ids": event.source_ids},
            source_id=source_id,
        )
        graph.add_memory(
            memory_id=event.id,
            content=f"{event.content}\n[event_occurrence={event.sequence}]",
            source_refs=(source_id,),
            embedding=event.embedding,
            now=float(event.sequence),
        )
    events = tuple(organization.events.values())
    sequence = {edge: 1.0 for edge in organization.sequence_edges}
    semantic = {
        (left.id, right.id): similarity
        for left_index, left in enumerate(events)
        for right in events[left_index + 1 :]
        for similarity in (
            max(0.0, cosine_similarity(left.embedding, right.embedding)),
        )
        if similarity > 0.0
    }
    graph.replace_relations(_combine_layers(sequence, semantic))
    return (hierarchy_builder or StructuralEntropyHierarchyBuilder()).build(graph)


def smallest_local_event_ids(
    hierarchy: MemoryHierarchy, seed_event_ids: Sequence[str]
) -> tuple[str, ...]:
    paths = _cluster_paths(hierarchy)
    seeds = tuple(dict.fromkeys(seed_event_ids))
    expanded = list(seeds)
    for seed in seeds:
        path = paths.get(seed, ())
        local = next(
            (
                hierarchy.clusters[cluster_id].member_ids
                for cluster_id in reversed(path)
                if len(hierarchy.clusters[cluster_id].member_ids) > 1
            ),
            (seed,),
        )
        expanded.extend(
            (member for member in sorted(local, reverse=True) if member not in seeds)
        )
    return tuple(dict.fromkeys(expanded))


def _cluster_paths(hierarchy: MemoryHierarchy) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for cluster_id, cluster in hierarchy.clusters.items():
        if cluster.child_ids:
            continue
        reverse_path = []
        current: str | None = cluster_id
        while current is not None:
            reverse_path.append(current)
            current = hierarchy.clusters[current].parent_id
        path = tuple(reversed(reverse_path))
        for event_id in cluster.member_ids:
            result[event_id] = path
    return result


def _combine_layers(
    *layers: Mapping[tuple[str, str], float],
) -> dict[tuple[str, str], float]:
    combined: dict[tuple[str, str], float] = {}
    for layer in layers:
        positive = {
            tuple(sorted(edge)): float(weight)
            for edge, weight in layer.items()
            if edge[0] != edge[1] and float(weight) > 0.0
        }
        total = sum(positive.values())
        if total <= 0.0:
            continue
        for edge, weight in positive.items():
            combined[edge] = combined.get(edge, 0.0) + weight / total
    return combined


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys((str(value).strip() for value in values if str(value).strip()))
    )
