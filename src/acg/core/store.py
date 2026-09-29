from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from hashlib import sha1
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .schema import MemoryUnit, Relation, SourceKind, SourceRecord


class SourceStore:
    def __init__(self) -> None:
        self._records: dict[str, SourceRecord] = {}
        self._next_sequence = 0

    @property
    def records(self) -> Mapping[str, SourceRecord]:
        return MappingProxyType(self._records)

    def append(
        self,
        kind: SourceKind,
        payload: Any,
        *,
        timestamp: float | None = None,
        source_id: str | None = None,
    ) -> SourceRecord:
        record_id = source_id or f"src-{self._next_sequence + 1:06d}"
        if record_id in self._records:
            raise ValueError(f"Duplicate source id: {record_id}")
        record = SourceRecord(
            id=record_id,
            kind=kind,
            payload=_freeze_payload(payload),
            sequence=self._next_sequence,
            timestamp=timestamp,
        )
        self._records[record_id] = record
        self._next_sequence += 1
        return record

    def get(self, source_id: str) -> SourceRecord:
        try:
            return self._records[source_id]
        except KeyError as error:
            raise KeyError(f"Unknown source id: {source_id}") from error

    def restore_record(self, record: Mapping[str, Any]) -> SourceRecord:
        source_id = str(record.get("id") or "").strip()
        if not source_id:
            raise ValueError("Restored source record must have an id")
        if source_id in self._records:
            existing = self._records[source_id]
            if existing != _source_record_from_mapping(record):
                raise ValueError(
                    f"Restored source id conflicts with existing record: {source_id}"
                )
            return existing
        restored = _source_record_from_mapping(record)
        self._records[source_id] = restored
        self._next_sequence = max(self._next_sequence, restored.sequence + 1)
        return restored

    def require(self, source_refs: Iterable[str]) -> tuple[str, ...]:
        normalized = _unique_nonempty(source_refs, field_name="source_refs")
        unknown = [
            source_id for source_id in normalized if source_id not in self._records
        ]
        if unknown:
            raise ValueError(f"Unknown source refs: {unknown}")
        return normalized


class MemoryGraph:
    def __init__(self, sources: SourceStore) -> None:
        self.sources = sources
        self._units: dict[str, MemoryUnit] = {}
        self._memory_by_content: dict[str, str] = {}
        self._relations: dict[str, Relation] = {}
        self._next_memory_number = 1

    @property
    def units(self) -> Mapping[str, MemoryUnit]:
        return MappingProxyType(self._units)

    @property
    def relations(self) -> Mapping[str, Relation]:
        return MappingProxyType(self._relations)

    def add_memory(
        self,
        *,
        content: str,
        source_refs: Iterable[str],
        embedding: Iterable[float] | None = None,
        now: float | None = None,
        memory_id: str | None = None,
    ) -> MemoryUnit:
        normalized_content = _nonempty_text(content, field_name="content")
        normalized_refs = self.sources.require(source_refs)
        existing_id = self._memory_by_content.get(normalized_content)
        if existing_id is not None:
            if memory_id is not None and memory_id != existing_id:
                raise ValueError(
                    f"Memory content already belongs to {existing_id}, not {memory_id}"
                )
            current = self._units[existing_id]
            merged_refs = tuple(dict.fromkeys((*current.source_refs, *normalized_refs)))
            if merged_refs != current.source_refs:
                timestamp = time.time() if now is None else now
                current = replace(
                    current, source_refs=merged_refs, updated_at=timestamp
                )
                self._units[existing_id] = current
            return current
        unit_id = memory_id or self._allocate_memory_id()
        if unit_id in self._units:
            raise ValueError(f"Duplicate memory id: {unit_id}")
        timestamp = time.time() if now is None else now
        unit = MemoryUnit(
            id=unit_id,
            content=normalized_content,
            source_refs=normalized_refs,
            embedding=_normalize_embedding(embedding),
            created_at=timestamp,
            updated_at=timestamp,
        )
        self._units[unit_id] = unit
        self._memory_by_content[normalized_content] = unit_id
        return unit

    def add_relation(
        self,
        *,
        source: str,
        target: str,
        weight: float = 1.0,
        relation_id: str | None = None,
    ) -> Relation:
        if source not in self._units or target not in self._units:
            raise ValueError(f"Relation endpoints must exist: {source!r}, {target!r}")
        if source == target:
            raise ValueError("Relation cannot be a self-edge")
        normalized_weight = float(weight)
        if not math.isfinite(normalized_weight) or normalized_weight <= 0.0:
            raise ValueError("Relation weight must be finite and positive")
        left, right = sorted((source, target))
        existing = self.find_relation(source=left, target=right)
        if existing is not None:
            if relation_id is not None and relation_id != existing.id:
                raise ValueError(
                    f"Relation already exists as {existing.id}, not {relation_id}"
                )
            if normalized_weight != existing.weight:
                raise ValueError(
                    f"Relation {existing.id} already has weight {existing.weight}"
                )
            return existing
        stable_id = relation_id or _relation_id(left, right)
        if stable_id in self._relations:
            raise ValueError(f"Duplicate relation id: {stable_id}")
        relation = Relation(
            id=stable_id, source=left, target=right, weight=normalized_weight
        )
        self._relations[stable_id] = relation
        return relation

    def replace_relations(
        self, edge_weights: Mapping[tuple[str, str], float]
    ) -> tuple[str, ...]:
        replacement: dict[str, Relation] = {}
        for endpoints, weight in sorted(edge_weights.items()):
            if len(endpoints) != 2:
                raise ValueError(
                    f"Relation endpoints must contain two ids: {endpoints!r}"
                )
            left, right = sorted((str(endpoints[0]), str(endpoints[1])))
            if left not in self._units or right not in self._units:
                raise ValueError(f"Relation endpoints must exist: {left!r}, {right!r}")
            if left == right:
                raise ValueError("Relation cannot be a self-edge")
            normalized_weight = float(weight)
            if not math.isfinite(normalized_weight) or normalized_weight <= 0.0:
                raise ValueError("Relation weight must be finite and positive")
            relation_id = _relation_id(left, right)
            collided = replacement.get(relation_id)
            if collided is not None and {collided.source, collided.target} != {
                left,
                right,
            }:
                raise ValueError(
                    f"Relation id collision: {relation_id} maps to both {(collided.source, collided.target)!r} and {(left, right)!r}"
                )
            replacement[relation_id] = Relation(
                id=relation_id, source=left, target=right, weight=normalized_weight
            )
        self._relations = replacement
        return tuple(replacement)

    def get_memory(self, memory_id: str) -> MemoryUnit:
        try:
            return self._units[memory_id]
        except KeyError as error:
            raise KeyError(f"Unknown memory id: {memory_id}") from error

    def restore_unit(
        self, record: Mapping[str, Any], *, embedding: Iterable[float] | None = None
    ) -> MemoryUnit:
        memory_id = str(record.get("id") or "").strip()
        content = str(record.get("content") or "").strip()
        if not memory_id or not content:
            raise ValueError("Restored memory units require id and content")
        source_refs = tuple((str(item) for item in record.get("source_refs") or ()))
        self.sources.require(source_refs)
        existing = self._units.get(memory_id)
        if existing is not None:
            if existing.content != content or existing.source_refs != source_refs:
                raise ValueError(
                    f"Restored memory id conflicts with existing unit: {memory_id}"
                )
            return existing
        existing_content_id = self._memory_by_content.get(content)
        if existing_content_id is not None and existing_content_id != memory_id:
            raise ValueError(
                f"Restored memory content already belongs to {existing_content_id}, not {memory_id}"
            )
        created_at = float(record.get("created_at") or 0.0)
        updated_at = float(record.get("updated_at") or created_at)
        unit = MemoryUnit(
            id=memory_id,
            content=content,
            source_refs=source_refs,
            embedding=_normalize_embedding(embedding),
            created_at=created_at,
            updated_at=updated_at,
        )
        self._units[memory_id] = unit
        self._memory_by_content[content] = memory_id
        suffix = memory_id.removeprefix("mem-")
        if suffix.isdigit():
            self._next_memory_number = max(self._next_memory_number, int(suffix) + 1)
        return unit

    def find_memory_by_content(self, content: str) -> MemoryUnit | None:
        normalized_content = _nonempty_text(content, field_name="content")
        memory_id = self._memory_by_content.get(normalized_content)
        return self._units.get(memory_id) if memory_id is not None else None

    def get_relation(self, relation_id: str) -> Relation:
        try:
            return self._relations[relation_id]
        except KeyError as error:
            raise KeyError(f"Unknown relation id: {relation_id}") from error

    def set_embeddings(self, embeddings: Mapping[str, Iterable[float]]) -> None:
        unknown = sorted(set(embeddings) - set(self._units))
        if unknown:
            raise ValueError(f"Cannot index unknown memories: {unknown}")
        normalized = {
            memory_id: _normalize_embedding(vector)
            for memory_id, vector in embeddings.items()
        }
        for memory_id, vector in normalized.items():
            self._units[memory_id] = replace(self._units[memory_id], embedding=vector)

    def incident_relations(self, memory_id: str) -> tuple[Relation, ...]:
        self.get_memory(memory_id)
        return tuple(
            (
                relation
                for relation in self._relations.values()
                if relation.touches(memory_id)
            )
        )

    def neighbors(self, memory_id: str) -> tuple[str, ...]:
        neighbors = {
            relation.other(memory_id) for relation in self.incident_relations(memory_id)
        }
        return tuple(sorted(neighbors))

    def find_relation(self, *, source: str, target: str) -> Relation | None:
        endpoints = {source, target}
        for relation in self._relations.values():
            if {relation.source, relation.target} == endpoints:
                return relation
        return None

    def _allocate_memory_id(self) -> str:
        while (candidate := f"mem-{self._next_memory_number:06d}") in self._units:
            self._next_memory_number += 1
        self._next_memory_number += 1
        return candidate


def source_payload_text(record: SourceRecord) -> str:
    if isinstance(record.payload, str):
        return record.payload
    return json.dumps(_thaw_payload(record.payload), ensure_ascii=False, sort_keys=True)


def _source_record_from_mapping(record: Mapping[str, Any]) -> SourceRecord:
    try:
        kind = SourceKind(str(record.get("kind")))
    except ValueError as error:
        raise ValueError(
            f"Unknown restored source kind: {record.get('kind')!r}"
        ) from error
    sequence = record.get("sequence")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
        raise ValueError("Restored source sequence must be a non-negative integer")
    source_id = str(record.get("id") or "").strip()
    if not source_id:
        raise ValueError("Restored source record must have an id")
    return SourceRecord(
        id=source_id,
        kind=kind,
        payload=_freeze_payload(record.get("payload")),
        sequence=sequence,
        timestamp=record.get("timestamp"),
    )


def _freeze_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_payload(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple((_freeze_payload(item) for item in value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(
        f"Source payload must be JSON-compatible, got {type(value).__name__}"
    )


def _thaw_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_payload(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_payload(item) for item in value]
    return value


def _unique_nonempty(values: Iterable[str], *, field_name: str) -> tuple[str, ...]:
    normalized = tuple(
        dict.fromkeys((str(value).strip() for value in values if str(value).strip()))
    )
    if not normalized:
        raise ValueError(f"{field_name} must contain at least one value")
    return normalized


def _nonempty_text(value: str, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _normalize_embedding(value: Iterable[float] | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    vector = tuple((float(item) for item in value))
    if not vector or any((not math.isfinite(item) for item in vector)):
        raise ValueError("Embedding must be a non-empty finite vector")
    return vector


def _relation_id(left: str, right: str) -> str:
    digest = sha1(f"{left}\x00{right}".encode("utf-8")).hexdigest()[:12]
    return f"rel-sem-{digest}"
