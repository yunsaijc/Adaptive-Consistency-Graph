from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .projection import (
    TEMPORAL_CURRENT,
    TEMPORAL_EARLIER,
    TEMPORAL_LATEST,
    TEMPORAL_UNSCOPED,
    Projection,
    ProjectionEdge,
)


@dataclass(frozen=True)
class HistoryItem:
    memory_id: str
    temporal_stratum: str
    event_id: str | None
    object_scope_ids: tuple[str, ...]
    route_basis: str


class HistoryProvider(Protocol):
    name: str

    def order(
        self, projection: Projection, memory_ids: Sequence[str]
    ) -> tuple[HistoryItem, ...]:
        pass


class FlatHistoryProvider:
    name = "flat"

    def order(
        self, projection: Projection, memory_ids: Sequence[str]
    ) -> tuple[HistoryItem, ...]:
        return _items_for_memory_ids(projection, memory_ids, rank_by_input=True)


class EventStageHistoryProvider:
    name = "event_stage"

    def order(
        self, projection: Projection, memory_ids: Sequence[str]
    ) -> tuple[HistoryItem, ...]:
        items = _items_for_memory_ids(projection, memory_ids, rank_by_input=False)
        return tuple(
            sorted(
                items,
                key=lambda item: (
                    _TEMPORAL_RANK.get(item.temporal_stratum, 9),
                    -projection.event_sequences.get(item.event_id, -1),
                    item.memory_id,
                ),
            )
        )


_TEMPORAL_RANK = {
    TEMPORAL_CURRENT: 0,
    TEMPORAL_LATEST: 1,
    TEMPORAL_UNSCOPED: 2,
    TEMPORAL_EARLIER: 3,
}


def hierarchy_edges(
    projection: Projection, *, provider: HistoryProvider | None = None
) -> tuple[ProjectionEdge, ...]:
    history_provider = provider or EventStageHistoryProvider()
    memory_ids = tuple(
        dict.fromkeys((incidence.memory_id for incidence in projection.incidences))
    )
    ordered = history_provider.order(projection, memory_ids)
    edges = list(projection.edges)
    previous: str | None = None
    for item in ordered:
        current = f"evidence:{item.memory_id}"
        if previous is not None and previous != current:
            edges.append(
                ProjectionEdge(
                    parent_id=previous,
                    child_id=current,
                    kind="history_order",
                    provenance=history_provider.name,
                )
            )
        previous = current
    return tuple(
        sorted(
            {
                (edge.parent_id, edge.child_id, edge.kind): edge for edge in edges
            }.values(),
            key=lambda edge: (edge.parent_id, edge.child_id, edge.kind),
        )
    )


def _items_for_memory_ids(
    projection: Projection, memory_ids: Sequence[str], *, rank_by_input: bool
) -> tuple[HistoryItem, ...]:
    requested = tuple(dict.fromkeys(memory_ids))
    rank = {memory_id: index for index, memory_id in enumerate(requested)}
    by_memory: dict[str, list] = {}
    for incidence in projection.incidences:
        if incidence.memory_id in rank:
            by_memory.setdefault(incidence.memory_id, []).append(incidence)
    items: list[HistoryItem] = []
    for memory_id in requested:
        incidences = by_memory.get(memory_id, ())
        if not incidences:
            continue
        first = sorted(
            incidences,
            key=lambda incidence: (
                _TEMPORAL_RANK.get(incidence.temporal_stratum, 9),
                incidence.event_id or "",
                incidence.object_scope_id or "",
                incidence.requirement_id,
            ),
        )[0]
        items.append(
            HistoryItem(
                memory_id=memory_id,
                temporal_stratum=first.temporal_stratum,
                event_id=first.event_id,
                object_scope_ids=tuple(
                    sorted(
                        {
                            incidence.object_scope_id
                            for incidence in incidences
                            if incidence.object_scope_id is not None
                        }
                    )
                ),
                route_basis=first.route_basis,
            )
        )
    if rank_by_input:
        return tuple(sorted(items, key=lambda item: rank[item.memory_id]))
    return tuple(items)


__all__ = [
    "HistoryItem",
    "HistoryProvider",
    "EventStageHistoryProvider",
    "FlatHistoryProvider",
    "hierarchy_edges",
]
