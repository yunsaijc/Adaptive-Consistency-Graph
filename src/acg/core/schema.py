from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping


class SourceKind(StrEnum):
    TASK = "task"
    ACTION = "action"
    OBSERVATION = "observation"
    STATE = "state"


@dataclass(frozen=True)
class SourceRecord:
    id: str
    kind: SourceKind
    payload: Any
    sequence: int
    timestamp: float | None = None


@dataclass(frozen=True)
class MemoryUnit:
    id: str
    content: str
    source_refs: tuple[str, ...]
    embedding: tuple[float, ...] | None
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class Relation:
    id: str
    source: str
    target: str
    weight: float = 1.0

    def touches(self, memory_id: str) -> bool:
        return self.source == memory_id or self.target == memory_id

    def other(self, memory_id: str) -> str:
        if self.source == memory_id:
            return self.target
        if self.target == memory_id:
            return self.source
        raise ValueError(
            f"Memory {memory_id!r} is not an endpoint of relation {self.id!r}"
        )


@dataclass(frozen=True)
class AddMemory:
    temp_id: str
    content: str


@dataclass(frozen=True)
class GraphDelta:
    operations: tuple[AddMemory, ...] = ()


@dataclass(frozen=True)
class RejectedOperation:
    operation_index: int
    reason: str


@dataclass(frozen=True)
class AppliedDelta:
    added_memory_ids: tuple[str, ...] = ()
    reused_memory_ids: tuple[str, ...] = ()
    temp_id_map: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    rejected: tuple[RejectedOperation, ...] = ()

    @property
    def embedding_memory_ids(self) -> tuple[str, ...]:
        return self.added_memory_ids


@dataclass(frozen=True)
class HierarchyCluster:
    id: str
    parent_id: str | None
    member_ids: tuple[str, ...]
    child_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoryHierarchy:
    root_id: str
    clusters: Mapping[str, HierarchyCluster]
    entropy_bits: float = 0.0

    @property
    def root(self) -> HierarchyCluster:
        return self.clusters[self.root_id]

    def children_of(self, cluster_id: str) -> tuple[HierarchyCluster, ...]:
        cluster = self.clusters[cluster_id]
        return tuple((self.clusters[child_id] for child_id in cluster.child_ids))
