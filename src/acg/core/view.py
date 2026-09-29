from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .retrieval import RetrievedSubgraph
from .schema import MemoryUnit, Relation, SourceRecord
from .store import MemoryGraph, source_payload_text

TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class DecisionView:
    task: SourceRecord
    recent_sources: tuple[SourceRecord, ...]
    memories: tuple[MemoryUnit, ...]
    relations: tuple[Relation, ...]
    grounding_sources: tuple[SourceRecord, ...]
    retrieved_memory_ids: tuple[str, ...]
    earlier_memory_ids: tuple[str, ...]
    focused_recent_event_ids: tuple[str, ...]
    reference_recent_events_enabled: bool
    query: str
    ordinary_rendered: str
    memory_tokens: int
    projection: Any | None = None
    rendered: Any | None = None


def build_decision_view(
    graph: MemoryGraph,
    *,
    task: SourceRecord,
    recent_sources: Sequence[SourceRecord],
    retrieved: RetrievedSubgraph,
    available_memory_tokens: int | None,
    token_counter: TokenCounter | None = None,
    include_grounding_sources: bool = True,
    recent_event_ids_by_source: Mapping[str, str] | None = None,
    earlier_memory_ids: Sequence[str] = (),
) -> DecisionView:
    if available_memory_tokens is not None and available_memory_tokens < 0:
        raise ValueError("Available memory tokens must be non-negative or None")
    count_tokens = token_counter or approximate_token_count
    recent = tuple(recent_sources)
    visible_source_ids = {task.id, *(record.id for record in recent)}
    referencable_source_ids = set(recent_event_ids_by_source or ())
    excluded_memory_ids = {
        memory_id
        for memory_id, memory in graph.units.items()
        if memory.source_refs
        and set(memory.source_refs).issubset(visible_source_ids)
        and (
            not (
                recent_event_ids_by_source is not None
                and set(memory.source_refs).issubset(referencable_source_ids)
            )
        )
    }
    selected_ids = _fit_closure_groups(
        graph,
        retrieved,
        excluded_memory_ids=excluded_memory_ids,
        available_memory_tokens=available_memory_tokens,
        token_counter=count_tokens,
        recent_event_ids_by_source=recent_event_ids_by_source,
        earlier_memory_ids=set(earlier_memory_ids),
    )
    memory_ids, focused_recent_event_ids = _partition_selected_memory(
        graph, selected_ids, recent_event_ids_by_source=recent_event_ids_by_source
    )
    selected_set = set(memory_ids)
    memories = tuple((graph.units[memory_id] for memory_id in memory_ids))
    relations = tuple(
        (
            graph.relations[relation_id]
            for relation_id in retrieved.relation_ids
            if relation_id in graph.relations
            and graph.relations[relation_id].source in selected_set
            and (graph.relations[relation_id].target in selected_set)
        )
    )
    source_ids = tuple(
        sorted(
            {
                source_id
                for memory in memories
                for source_id in memory.source_refs
                if source_id not in visible_source_ids
            },
            key=lambda source_id: graph.sources.get(source_id).sequence,
        )
    )
    grounding_sources = (
        _fit_grounding_sources(
            memories,
            relations,
            tuple((graph.sources.get(source_id) for source_id in source_ids)),
            available_memory_tokens=available_memory_tokens,
            token_counter=count_tokens,
            focused_recent_event_ids=focused_recent_event_ids,
        )
        if include_grounding_sources
        else ()
    )
    rendered = render_decision_view_parts(
        task=task,
        recent_sources=recent,
        memories=memories,
        relations=relations,
        grounding_sources=grounding_sources,
        focused_recent_event_ids=focused_recent_event_ids,
        earlier_memory_ids=set(earlier_memory_ids),
    )
    memory_rendered = _render_memory_parts(
        memories,
        relations,
        grounding_sources,
        focused_recent_event_ids,
        earlier_memory_ids=set(earlier_memory_ids),
    )
    return DecisionView(
        task=task,
        recent_sources=recent,
        memories=memories,
        relations=relations,
        grounding_sources=grounding_sources,
        retrieved_memory_ids=selected_ids,
        earlier_memory_ids=tuple(
            (
                memory_id
                for memory_id in memory_ids
                if memory_id in set(earlier_memory_ids)
            )
        ),
        focused_recent_event_ids=focused_recent_event_ids,
        reference_recent_events_enabled=recent_event_ids_by_source is not None,
        query=retrieved.query,
        ordinary_rendered=rendered,
        memory_tokens=count_tokens(memory_rendered),
    )


def render_decision_view(view: DecisionView) -> str:
    return view.rendered.text if view.rendered is not None else view.ordinary_rendered


def render_memory_context(view: DecisionView) -> str:
    return _render_memory_parts(
        view.memories,
        view.relations,
        view.grounding_sources,
        view.focused_recent_event_ids,
        earlier_memory_ids=set(view.earlier_memory_ids),
    )


def render_decision_view_parts(
    *,
    task: SourceRecord,
    recent_sources: Sequence[SourceRecord],
    memories: Sequence[MemoryUnit],
    relations: Sequence[Relation],
    grounding_sources: Sequence[SourceRecord],
    focused_recent_event_ids: Sequence[str] = (),
    earlier_memory_ids: set[str] | None = None,
) -> str:
    sections = [f"[Original task]\n{source_payload_text(task)}"]
    if recent_sources:
        sections.append(
            "[Recent interaction]\n"
            + "\n".join(
                (
                    f"[{record.id} | {record.kind.value}] {source_payload_text(record)}"
                    for record in recent_sources
                )
            )
        )
    memory = _render_memory_parts(
        memories,
        relations,
        grounding_sources,
        focused_recent_event_ids,
        earlier_memory_ids=earlier_memory_ids,
    )
    if memory:
        sections.append(memory)
    return "\n\n".join(sections)


def approximate_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _fit_closure_groups(
    graph: MemoryGraph,
    retrieved: RetrievedSubgraph,
    *,
    excluded_memory_ids: set[str],
    available_memory_tokens: int | None,
    token_counter: TokenCounter,
    recent_event_ids_by_source: Mapping[str, str] | None,
    earlier_memory_ids: set[str],
) -> tuple[str, ...]:
    if available_memory_tokens is None:
        return tuple(
            (
                memory_id
                for memory_id in retrieved.memory_ids
                if memory_id not in excluded_memory_ids
            )
        )
    selected: list[str] = []
    selected_set: set[str] = set()
    for group in retrieved.closure_groups:
        candidate_ids = [
            memory_id
            for memory_id in group
            if memory_id not in selected_set and memory_id not in excluded_memory_ids
        ]
        if not candidate_ids:
            continue
        proposed = (*selected, *candidate_ids)
        rendered = _render_selected_memory(
            graph,
            proposed,
            retrieved.relation_ids,
            recent_event_ids_by_source=recent_event_ids_by_source,
            earlier_memory_ids=earlier_memory_ids,
        )
        if token_counter(rendered) <= available_memory_tokens:
            selected.extend(candidate_ids)
            selected_set.update(candidate_ids)
    return tuple(selected)


def _render_selected_memory(
    graph: MemoryGraph,
    memory_ids: Sequence[str],
    relation_ids: Sequence[str],
    *,
    recent_event_ids_by_source: Mapping[str, str] | None,
    earlier_memory_ids: set[str],
) -> str:
    rendered_ids, focused_recent_event_ids = _partition_selected_memory(
        graph, memory_ids, recent_event_ids_by_source=recent_event_ids_by_source
    )
    selected = set(rendered_ids)
    memories = tuple((graph.units[memory_id] for memory_id in rendered_ids))
    relations = tuple(
        (
            graph.relations[relation_id]
            for relation_id in relation_ids
            if relation_id in graph.relations
            and graph.relations[relation_id].source in selected
            and (graph.relations[relation_id].target in selected)
        )
    )
    return _render_memory_parts(
        memories,
        relations,
        (),
        focused_recent_event_ids,
        earlier_memory_ids=earlier_memory_ids,
    )


def _partition_selected_memory(
    graph: MemoryGraph,
    memory_ids: Sequence[str],
    *,
    recent_event_ids_by_source: Mapping[str, str] | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if recent_event_ids_by_source is None:
        return (tuple(memory_ids), ())
    recent_sources = set(recent_event_ids_by_source)
    rendered_ids: list[str] = []
    focused_event_ids: list[str] = []
    for memory_id in memory_ids:
        memory = graph.units[memory_id]
        source_refs = set(memory.source_refs)
        if source_refs and source_refs.issubset(recent_sources):
            focused_event_ids.extend(
                (
                    recent_event_ids_by_source[source_id]
                    for source_id in memory.source_refs
                )
            )
        else:
            rendered_ids.append(memory_id)
    return (tuple(rendered_ids), tuple(dict.fromkeys(focused_event_ids)))


def _fit_grounding_sources(
    memories: Sequence[MemoryUnit],
    relations: Sequence[Relation],
    candidates: Sequence[SourceRecord],
    *,
    available_memory_tokens: int | None,
    token_counter: TokenCounter,
    focused_recent_event_ids: Sequence[str] = (),
) -> tuple[SourceRecord, ...]:
    if available_memory_tokens is None:
        return tuple(candidates)
    selected: list[SourceRecord] = []
    for source in candidates:
        proposed = (*selected, source)
        rendered = _render_memory_parts(
            memories, relations, proposed, focused_recent_event_ids
        )
        if token_counter(rendered) <= available_memory_tokens:
            selected.append(source)
    return tuple(selected)


def _render_memory_parts(
    memories: Sequence[MemoryUnit],
    relations: Sequence[Relation],
    grounding_sources: Sequence[SourceRecord],
    focused_recent_event_ids: Sequence[str] = (),
    earlier_memory_ids: set[str] | None = None,
) -> str:
    if not memories and (not focused_recent_event_ids):
        return ""
    sections = []
    if focused_recent_event_ids:
        sections.append(
            "[Graph-focused history]\n" + ", ".join(focused_recent_event_ids)
        )
    if memories:
        earlier = earlier_memory_ids or set()
        current_memories = tuple((unit for unit in memories if unit.id not in earlier))
        historical_memories = tuple((unit for unit in memories if unit.id in earlier))
        if not historical_memories:
            sections.append(
                "[Relevant long-term memory]\n" + _render_memories(current_memories)
            )
        elif current_memories:
            sections.append(
                "[More recent long-term memory]\n" + _render_memories(current_memories)
            )
        if historical_memories:
            sections.append(
                "[Earlier long-term memory]\n" + _render_memories(historical_memories)
            )
    if relations:
        sections.append(
            "[Semantic memory links]\n"
            + "\n".join(
                (
                    f"[{relation.id}] {relation.source} <-> {relation.target}"
                    for relation in relations
                )
            )
        )
    if grounding_sources:
        sections.append(
            "[Grounding sources]\n"
            + "\n".join(
                (
                    f"[{record.id} | {record.kind.value}] {source_payload_text(record)}"
                    for record in grounding_sources
                )
            )
        )
    return "\n\n".join(sections)


def _render_memories(memories: Sequence[MemoryUnit]) -> str:
    return "\n".join(
        (
            f"[{unit.id} | sources={','.join(unit.source_refs)}] {unit.content}"
            for unit in memories
        )
    )
