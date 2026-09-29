from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .builder import BuilderProposal, MemoryBuilder, write_complete_evidence
from .concepts import (
    ConceptContext,
    ConceptIndexer,
    ConceptProposal,
    apply_concept_proposal,
    apply_exact_concept_memberships,
    concept_candidates,
    concept_records,
    concepts_missing_embeddings,
)
from .efficient_hierarchy import EfficientStructuralEntropyHierarchyBuilder
from .hierarchy import HierarchyBuilder
from .organization import (
    EventOrganization,
    build_event_hierarchy,
    smallest_local_event_ids,
)
from .relations import rebuild_semantic_relations
from .retrieval import EmbeddingProvider, index_memories, retrieve_memory_branches
from .schema import (
    AppliedDelta,
    GraphDelta,
    MemoryHierarchy,
    MemoryUnit,
    SourceKind,
    SourceRecord,
)
from .store import MemoryGraph, SourceStore, source_payload_text
from .view import DecisionView, TokenCounter, build_decision_view


@dataclass(frozen=True)
class ACGConfig:
    retrieval_top_k: int = 8
    builder_context_top_k: int = 8
    closure_depth: int = 1
    hierarchy_route_width: int = 2
    concept_candidate_top_k: int = 8
    projection_builder_retry_max_tokens: int = 4096
    projection_tokens: int = 5000

    def __post_init__(self) -> None:
        if self.retrieval_top_k <= 0 or self.builder_context_top_k <= 0:
            raise ValueError("Retrieval limits must be positive")
        if self.closure_depth < 0:
            raise ValueError("Closure depth must be non-negative")
        if self.hierarchy_route_width <= 0:
            raise ValueError("Hierarchy route width must be positive")
        if self.concept_candidate_top_k <= 0:
            raise ValueError("Concept candidate limit must be positive")
        if self.projection_builder_retry_max_tokens <= 0:
            raise ValueError("Builder retry token limit must be positive")
        if self.projection_tokens < 0:
            raise ValueError("Projection token limit must be non-negative")


@dataclass(frozen=True)
class GraphUpdateResult:
    source_ids: tuple[str, ...]
    proposal: BuilderProposal
    applied: AppliedDelta
    hierarchy: MemoryHierarchy
    event_id: str | None = None
    concept_proposal: ConceptProposal = ConceptProposal()
    skipped: bool = False


def _preserve_render_failure(owner: Any, projection: Any, error: Exception) -> None:
    owner.failed_projection = projection
    owner.render_failure = {"type": type(error).__name__, "message": str(error)}


class AdaptiveConsistencyGraph:
    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        builder: MemoryBuilder | None = None,
        concept_indexer: ConceptIndexer | None = None,
        hierarchy_builder: HierarchyBuilder | None = None,
        config: ACGConfig | None = None,
    ) -> None:
        self.builder = builder
        self.concept_indexer = concept_indexer
        self.embedding_provider = embedding_provider
        self.hierarchy_builder = (
            hierarchy_builder or EfficientStructuralEntropyHierarchyBuilder()
        )
        self.config = config or ACGConfig()
        self.sources = SourceStore()
        self.graph = MemoryGraph(self.sources)
        self.hierarchy = self.hierarchy_builder.build(self.graph)
        self.organization = EventOrganization()
        self.event_hierarchy = build_event_hierarchy(self.organization)
        self._history_hierarchy_by_event: dict[str, MemoryHierarchy] = {}
        self._task_source_id: str | None = None
        self._processed_source_ids: set[str] = set()
        self.projection_source_state: Any | None = None
        self.lineage: Any | None = None
        self._projection_last_action_source_id: str | None = None
        self._projection_evidence_extractor: Any | None = None
        self._projection_requirement_vector_key: tuple[tuple[Any, ...], ...] | None = (
            None
        )
        self._projection_requirement_vectors: tuple[tuple[float, ...], ...] = ()
        self.failed_projection: Any | None = None
        self.render_failure: Mapping[str, str] | None = None
        self._enable_projection_runtime()

    @property
    def task(self) -> SourceRecord:
        if self._task_source_id is None:
            raise RuntimeError("ACG  has not been started with a task")
        return self.sources.get(self._task_source_id)

    @property
    def processed_source_ids(self) -> frozenset[str]:
        return frozenset(self._processed_source_ids)

    def start(
        self, task: Any, *, timestamp: float | None = None, source_id: str | None = None
    ) -> SourceRecord:
        if self._task_source_id is not None:
            raise RuntimeError("ACG  already has a task")
        record = self.sources.append(
            SourceKind.TASK, task, timestamp=timestamp, source_id=source_id
        )
        self._task_source_id = record.id
        self._start_projection_runtime(record)
        return record

    def restore_projection_payload(
        self,
        *,
        task_source: Mapping[str, Any],
        requirements: Sequence[Mapping[str, Any]],
        sources: Mapping[str, Mapping[str, Any]],
        memories: Mapping[str, Mapping[str, Any]],
        events: Mapping[str, Mapping[str, Any]],
        concepts: Mapping[str, Mapping[str, Any]],
        lineage: Mapping[str, Any],
    ) -> None:
        if self._task_source_id is not None:
            raise RuntimeError("ACG  already has a task")
        task_record = self.sources.restore_record(task_source)
        if task_record.kind != SourceKind.TASK:
            raise ValueError("Projection payload task source must have kind=task")
        self._task_source_id = task_record.id
        for record in sorted(
            sources.values(), key=lambda item: int(item.get("sequence", 0))
        ):
            self.sources.restore_record(record)
        from .lineage import RequirementLineageIndex
        from .source_context import (
            register_task_contract,
            register_task_source,
            source_grounded_requirement_anchors,
        )

        register_task_source(
            self, source_payload_text(task_record), source_id=task_record.id
        )
        register_task_contract(self, requirements)
        state = self.projection_source_state
        if state is not None:
            state.task_contract_status = {
                "status": "registered",
                "source": "restored_projection_payload",
                "n_requirements": len(requirements),
            }
        for record in sorted(memories.values(), key=lambda item: str(item.get("id"))):
            self.graph.restore_unit(record)
        memory_ids = tuple(self.graph.units)
        if memory_ids:
            index_memories(self.graph, self.embedding_provider, memory_ids)
            rebuild_semantic_relations(self.graph)
        event_records = sorted(
            events.values(), key=lambda item: int(item.get("sequence", 0))
        )
        if event_records:
            event_vectors = self.embedding_provider.embed(
                [str(item.get("content") or "") for item in event_records]
            )
            if len(event_vectors) != len(event_records):
                raise RuntimeError(
                    "Embedding provider returned the wrong number of event vectors"
                )
            for record, vector in zip(event_records, event_vectors):
                self.organization.restore_event(record, embedding=vector)
        concept_records = sorted(
            concepts.values(), key=lambda item: str(item.get("id"))
        )
        if concept_records:
            concept_vectors = self.embedding_provider.embed(
                [
                    "\n".join((str(value) for value in item.get("anchor_texts") or ()))
                    for item in concept_records
                ]
            )
            if len(concept_vectors) != len(concept_records):
                raise RuntimeError(
                    "Embedding provider returned the wrong number of concept vectors"
                )
            for record, vector in zip(concept_records, concept_vectors):
                self.organization.restore_concept(
                    record,
                    embedding=vector,
                    active_event_ids=record.get("event_ids") or (),
                )
        self.hierarchy = self.hierarchy_builder.build(self.graph)
        self.event_hierarchy = build_event_hierarchy(self.organization)
        self._history_hierarchy_by_event = {
            event.id: self.event_hierarchy
            for event in self.organization.events.values()
        }
        self._processed_source_ids = {
            record.id
            for record in self.sources.records.values()
            if record.kind != SourceKind.TASK
        }
        actions = [
            record
            for record in self.sources.records.values()
            if record.kind == SourceKind.ACTION
        ]
        self._projection_last_action_source_id = (
            max(actions, key=lambda item: item.sequence).id if actions else None
        )
        if self.lineage is None:
            self.lineage = RequirementLineageIndex()
        self.lineage.restore_snapshot(
            lineage,
            requirements=source_grounded_requirement_anchors(self),
            sources=self.sources.records,
        )

    def record(
        self,
        kind: SourceKind,
        payload: Any,
        *,
        timestamp: float | None = None,
        source_id: str | None = None,
    ) -> SourceRecord:
        _ = self.task
        if kind == SourceKind.TASK:
            raise ValueError("Use start() to record the single original task")
        record = self.sources.append(
            kind, payload, timestamp=timestamp, source_id=source_id
        )
        self._record_projection_source(record)
        return record

    def remember_exact(self, *, content: str, source_refs: Sequence[str]) -> MemoryUnit:
        _ = self.task
        existing = self.graph.find_memory_by_content(content)
        unit = self.graph.add_memory(content=content, source_refs=source_refs)
        if existing is None:
            index_memories(self.graph, self.embedding_provider, (unit.id,))
            rebuild_semantic_relations(self.graph)
        if existing is None or unit.source_refs != existing.source_refs:
            self.hierarchy = self.hierarchy_builder.build(self.graph)
        return self.graph.get_memory(unit.id)

    def update(self, source_ids: Sequence[str]) -> GraphUpdateResult:
        task = self.task
        unique_ids = tuple(dict.fromkeys(source_ids))
        if task.id in unique_ids:
            raise ValueError("The original task is a read-only relevance reference")
        records = tuple(
            sorted(
                (
                    self.sources.get(source_id)
                    for source_id in unique_ids
                    if source_id not in self._processed_source_ids
                ),
                key=lambda record: record.sequence,
            )
        )
        if not records:
            return GraphUpdateResult(
                source_ids=(),
                proposal=BuilderProposal(delta=GraphDelta()),
                applied=AppliedDelta(),
                hierarchy=self.hierarchy,
                skipped=True,
            )
        complete_write = write_complete_evidence(
            self.graph, records, evidence_extractor=self._projection_evidence_extractor
        )
        applied = complete_write.applied
        proposal = BuilderProposal(delta=GraphDelta())
        embedding_ids = tuple(
            (
                memory_id
                for memory_id in applied.embedding_memory_ids
                if memory_id in self.graph.units
            )
        )
        event_text = _event_text(records)
        embedding_texts = [
            *(self.graph.units[memory_id].content for memory_id in embedding_ids),
            event_text,
        ]
        event_vectors = self.embedding_provider.embed(embedding_texts)
        if len(event_vectors) != len(embedding_texts):
            raise RuntimeError(
                "Embedding provider returned the wrong number of vectors"
            )
        if embedding_ids:
            self.graph.set_embeddings(dict(zip(embedding_ids, event_vectors[:-1])))
        if applied.added_memory_ids:
            rebuild_semantic_relations(self.graph)
        if applied.added_memory_ids or applied.reused_memory_ids:
            self.hierarchy = self.hierarchy_builder.build(self.graph)
        event = self.organization.add_event(
            source_ids=(record.id for record in records),
            evidence_ids=(
                memory_id
                for record in records
                for memory_id, memory in self.graph.units.items()
                if record.id in memory.source_refs
            ),
            content=event_text,
            embedding=event_vectors[-1],
        )
        self._history_hierarchy_by_event[event.id] = self.event_hierarchy
        concept_proposal = self._index_event_concepts(event.id, records)
        self.event_hierarchy = build_event_hierarchy(self.organization)
        self._processed_source_ids.update((record.id for record in records))
        self._finalize_projection_update(
            result_source_ids=tuple((record.id for record in records)),
            result_raw_output=concept_proposal.raw_output,
        )
        return GraphUpdateResult(
            source_ids=tuple((record.id for record in records)),
            proposal=proposal,
            applied=applied,
            hierarchy=self.hierarchy,
            event_id=event.id,
            concept_proposal=concept_proposal,
        )

    def decision_view(
        self,
        *,
        recent_source_ids: Sequence[str],
        latest_action_source_id: str | None = None,
        latest_observation_source_id: str | None = None,
        available_memory_tokens: int | None,
        token_counter: TokenCounter | None = None,
        excluded_source_ids: Sequence[str] = (),
    ) -> DecisionView:
        task = self.task
        recent = tuple(
            sorted(
                (
                    self.sources.get(source_id)
                    for source_id in dict.fromkeys(recent_source_ids)
                ),
                key=lambda record: record.sequence,
            )
        )
        latest_action = _resolve_latest(
            self.sources,
            explicit_id=latest_action_source_id,
            recent=recent,
            kind=SourceKind.ACTION,
        )
        latest_observation = _resolve_latest(
            self.sources,
            explicit_id=latest_observation_source_id,
            recent=recent,
            kind=SourceKind.OBSERVATION,
        )
        event_records = tuple(
            (
                record
                for record in (latest_action, latest_observation)
                if record is not None
            )
        )
        event_query = _event_text(event_records)
        retrieved = retrieve_memory_branches(
            self.graph,
            queries=(event_query, source_payload_text(task)),
            embedding_provider=self.embedding_provider,
            hierarchy=self.hierarchy,
            top_k=self.config.retrieval_top_k,
            closure_depth=self.config.closure_depth,
        )
        current_event = next(
            (
                self.organization.event_for_source(record.id)
                for record in reversed(event_records)
                if self.organization.event_for_source(record.id) is not None
            ),
            None,
        )
        if current_event is not None:
            seed_events = self.organization.prior_events_for(current_event.id)
            local_events = smallest_local_event_ids(
                self.history_hierarchy_for(current_event.id), seed_events
            )
            retrieved = _supplement_retrieved(
                self.graph,
                retrieved,
                self.organization.evidence_ids_for_events(local_events),
            )
        view = build_decision_view(
            self.graph,
            task=task,
            recent_sources=recent,
            retrieved=retrieved,
            available_memory_tokens=available_memory_tokens,
            token_counter=token_counter,
            include_grounding_sources=False,
            recent_event_ids_by_source={
                record.id: event.id
                for record in recent
                for event in (self.organization.event_for_source(record.id),)
                if event is not None
            },
            earlier_memory_ids=self.organization.earlier_evidence_ids(current_event.id)
            if current_event is not None
            else (),
        )
        return self._attach_projection_view(
            view,
            current_event=current_event,
            available_memory_tokens=available_memory_tokens,
            token_counter=token_counter,
            excluded_source_ids=excluded_source_ids,
        )

    def snapshot(self) -> dict[str, Any]:
        from .source_context import source_grounded_requirement_anchors

        lineage_sources = {
            source_id: list(items)
            for source_id, items in getattr(self.lineage, "_by_source", {}).items()
        }
        return _jsonable(
            {
                "schema_version": "acg.core.v1",
                "config": self.config,
                "task_source_id": self._task_source_id,
                "sources": list(self.sources.records.values()),
                "memories": list(self.graph.units.values()),
                "relations": list(self.graph.relations.values()),
                "events": list(self.organization.events.values()),
                "concepts": list(self.organization.concepts.values()),
                "requirements": list(source_grounded_requirement_anchors(self)),
                "lineage": {
                    "schema_version": "acg.lineage.v1",
                    "sources": lineage_sources,
                    "telemetry": dict(getattr(self.lineage, "telemetry", {})),
                },
                "failed_projection": self.failed_projection,
            }
        )

    def lookup_source(self, source_id: str) -> SourceRecord:
        return self.sources.get(source_id)

    def history_hierarchy_for(self, event_id: str) -> MemoryHierarchy:
        try:
            return self._history_hierarchy_by_event[event_id]
        except KeyError as error:
            raise ValueError(f"Unknown event history hierarchy: {event_id}") from error

    def _index_event_concepts(
        self, event_id: str, records: Sequence[SourceRecord]
    ) -> ConceptProposal:
        if self.concept_indexer is None:
            return ConceptProposal()
        local_records = concept_records(
            records, evidence_extractor=self._projection_evidence_extractor
        )
        if not local_records:
            return ConceptProposal()
        evidence_ids_by_record = {
            record.id: (memory.id,)
            for record in local_records
            for content in ("\n".join((unit.text for unit in record.units)),)
            if content.strip()
            for memory in (self.graph.find_memory_by_content(content),)
            if memory is not None
        }
        event = self.organization.events[event_id]
        candidates = concept_candidates(
            self.organization,
            records=local_records,
            query_embedding=event.embedding,
            top_k=self.config.concept_candidate_top_k,
        )
        proposal = self.concept_indexer.propose(
            ConceptContext(
                task=self.task,
                records=local_records,
                candidate_ids=candidates,
                organization=self.organization,
            )
        )
        exact = apply_exact_concept_memberships(
            self.organization,
            event_id=event_id,
            records=local_records,
            evidence_ids_by_record=evidence_ids_by_record,
        )
        attached = apply_concept_proposal(
            self.organization,
            proposal,
            event_id=event_id,
            records=local_records,
            evidence_ids_by_record=evidence_ids_by_record,
        )
        missing = concepts_missing_embeddings(self.organization, (*exact, *attached))
        if missing:
            vectors = self.embedding_provider.embed(
                [self.organization.concepts[concept_id].text for concept_id in missing]
            )
            if len(vectors) != len(missing):
                raise RuntimeError(
                    "Embedding provider returned the wrong number of concept vectors"
                )
            self.organization.set_concept_embeddings(dict(zip(missing, vectors)))
        return proposal

    def _enable_projection_runtime(self) -> None:
        from .guard import ConceptIndexerGuard
        from .lineage import RequirementLineageIndex
        from .record_evidence import extract_record_level_evidence_records
        from .source_context import (
            pending_later_user_evidence,
            source_grounded_requirement_anchors,
        )

        self.lineage = RequirementLineageIndex()
        self._projection_evidence_extractor = extract_record_level_evidence_records
        if self.concept_indexer is None:
            return
        if not isinstance(self.concept_indexer, ConceptIndexerGuard):
            self.concept_indexer = ConceptIndexerGuard(
                self.concept_indexer,
                truncation_retry_max_tokens=self.config.projection_builder_retry_max_tokens,
            )
        guard = self.concept_indexer
        guard.set_origin_requirements_provider(
            lambda: source_grounded_requirement_anchors(self)
        )
        guard.set_later_user_sources_provider(lambda: pending_later_user_evidence(self))
        guard.set_source_payload_provider(
            lambda source_id: getattr(
                self.sources.records.get(str(source_id)), "payload", None
            )
        )
        guard.set_nearest_assistant_context_provider(
            self._projection_nearest_assistant_context
        )
        guard.set_dynamic_requirements_handler(
            self._apply_projection_dynamic_requirements
        )

    def _start_projection_runtime(self, record: SourceRecord) -> None:
        from .source_context import (
            register_task_contract_from_client,
            register_task_source,
            source_grounded_requirement_anchors,
        )

        register_task_source(self, source_payload_text(record))
        indexer = getattr(self, "concept_indexer", None)
        register_task_contract_from_client(self, getattr(indexer, "client", None))
        if self.lineage is not None:
            self.lineage.set_requirements(source_grounded_requirement_anchors(self))

    def _record_projection_source(self, record: SourceRecord) -> None:
        if self.lineage is None:
            return
        if record.kind == SourceKind.ACTION:
            from .lineage import action_origin_contract

            self.lineage.record_action_origins(record.id, record.payload)
            self.lineage.record_action_contract(
                record.id, action_origin_contract(record.payload)
            )
            self._projection_last_action_source_id = record.id
        elif record.kind == SourceKind.OBSERVATION:
            self.lineage.inherit_observation_origins(
                record.id,
                record.payload,
                parent_action_source_id=None
                if self.lineage.is_later_user_source(record.id)
                else self._projection_last_action_source_id,
            )

    def _finalize_projection_update(
        self, *, result_source_ids: Sequence[str], result_raw_output: Any
    ) -> None:
        if self.lineage is None:
            return
        indexer = getattr(self, "concept_indexer", None)
        apply_dynamic = getattr(indexer, "apply_dynamic_requirements", None)
        if callable(apply_dynamic):
            apply_dynamic(result_raw_output)
        if not isinstance(result_raw_output, Mapping):
            return
        if not any(
            (
                key in result_raw_output
                for key in ("action_origins", "user_source_decisions")
            )
        ):
            return
        for source_id in result_source_ids:
            source = self.sources.get(str(source_id))
            if source.kind == SourceKind.ACTION:
                self.lineage.record_structured_action_origins(
                    source.id, result_raw_output
                )

    def _attach_projection_view(
        self,
        view: DecisionView,
        *,
        current_event: Any | None,
        available_memory_tokens: int | None,
        token_counter: Any | None,
        excluded_source_ids: Sequence[str] = (),
    ) -> DecisionView:
        from .history import EventStageHistoryProvider
        from .projection import build_projection
        from .renderer import RenderBudgetError, render_projection
        from .source_context import source_grounded_requirement_anchors

        requirements = tuple(source_grounded_requirement_anchors(self))
        vectors = self._projection_requirement_vectors_for(requirements)
        effective_excluded_source_ids = tuple(
            dict.fromkeys(
                (
                    *excluded_source_ids,
                    *(current_event.source_ids if current_event is not None else ()),
                )
            )
        )
        projection = build_projection(
            self,
            view,
            requirements=requirements,
            current_event_id=current_event.id if current_event is not None else None,
            requirement_vectors=vectors,
            lineage_index=self.lineage,
            excluded_source_ids=effective_excluded_source_ids,
        )
        provider = EventStageHistoryProvider()
        budget = self.config.projection_tokens
        if available_memory_tokens is not None:
            budget = min(budget, max(0, int(available_memory_tokens)))
        projection = replace(projection, hierarchy_provider=provider.name)
        try:
            rendered = (
                render_projection(
                    projection,
                    max_tokens=budget,
                    token_counter=token_counter,
                    history_provider=provider,
                )
                if budget > 0
                else None
            )
        except RenderBudgetError as error:
            _preserve_render_failure(self, projection, error)
            raise
        self.failed_projection = None
        self.render_failure = None
        return replace(view, projection=projection, rendered=rendered)

    def _projection_requirement_vectors_for(
        self, requirements: Sequence[Any]
    ) -> tuple[tuple[float, ...], ...]:
        key = tuple(
            (
                (
                    item.id,
                    item.source_id,
                    item.start_offset,
                    item.end_offset,
                    item.exact_text,
                )
                for item in requirements
            )
        )
        if key == self._projection_requirement_vector_key:
            return self._projection_requirement_vectors
        vectors = tuple(
            (
                tuple(vector)
                for vector in self.embedding_provider.embed(
                    tuple((item.exact_text for item in requirements))
                )
            )
        )
        if len(vectors) != len(requirements):
            raise RuntimeError(
                "Embedding provider returned the wrong number of requirement vectors"
            )
        self._projection_requirement_vector_key = key
        self._projection_requirement_vectors = vectors
        return vectors

    def _projection_nearest_assistant_context(self) -> Mapping[str, Any] | None:
        candidates = [
            source
            for source in self.sources.records.values()
            if source.kind in {SourceKind.ACTION, SourceKind.STATE}
        ]
        if not candidates:
            return None
        source = max(candidates, key=lambda item: item.sequence)
        return {"source_id": source.id, "text": source_payload_text(source)}

    def _apply_projection_dynamic_requirements(
        self, entries: Sequence[Mapping[str, Any]]
    ) -> tuple[Any, ...]:
        from .source_context import (
            append_dynamic_requirements,
            pending_later_user_evidence,
            resolve_later_user_source,
            source_grounded_requirement_anchors,
        )

        by_source = {str(entry["source_id"]): entry for entry in entries}
        pending = tuple(
            sorted(pending_later_user_evidence(self), key=lambda item: item.sequence)
        )
        if set(by_source) != {item.source_id for item in pending}:
            raise ValueError(
                "User-source decision batch is not a complete pending-source closure"
            )
        appended: list[Any] = []
        for user_source in pending:
            segments = tuple(by_source[user_source.source_id]["segments"])
            decisions = {segment["decision"] for segment in segments}
            new_spans = tuple(
                (
                    segment["exact_span"]
                    for segment in segments
                    if segment["decision"] == "new_requirement"
                )
            )
            new_items = (
                append_dynamic_requirements(
                    self, source_id=user_source.source_id, exact_spans=new_spans
                )
                if new_spans
                else ()
            )
            appended.extend(new_items)
            decision = (
                "new_requirement"
                if decisions == {"new_requirement"}
                else "continuation_evidence"
                if decisions == {"continuation_evidence"}
                else "mixed"
            )
            continuation_ids = tuple(
                dict.fromkeys(
                    (
                        requirement_id
                        for segment in segments
                        for requirement_id in segment["continuation_requirement_ids"]
                    )
                )
            )
            resolve_later_user_source(
                self,
                source_id=user_source.source_id,
                decision=decision,
                exact_spans=tuple((segment["exact_span"] for segment in segments)),
                continuation_requirement_ids=continuation_ids,
                segments=segments,
            )
            if self.lineage is not None:
                self.lineage.record_user_source_decision(
                    user_source.source_id,
                    decision=decision,
                    exact_spans=tuple((segment["exact_span"] for segment in segments)),
                    continuation_requirement_ids=continuation_ids,
                    appended_requirement_ids=tuple((item.id for item in new_items)),
                    segments=segments,
                )
        if self.lineage is not None:
            self.lineage.set_requirements(source_grounded_requirement_anchors(self))
        return tuple(appended)


def _resolve_latest(
    sources: SourceStore,
    *,
    explicit_id: str | None,
    recent: Sequence[SourceRecord],
    kind: SourceKind,
) -> SourceRecord | None:
    if explicit_id is not None:
        record = sources.get(explicit_id)
        if record.kind != kind:
            raise ValueError(
                f"Source {explicit_id} is {record.kind.value}, expected {kind.value}"
            )
        return record
    return next((record for record in reversed(recent) if record.kind == kind), None)


def _event_text(records: Sequence[SourceRecord]) -> str:
    actions = [record for record in records if record.kind == SourceKind.ACTION]
    observations = [
        record for record in records if record.kind == SourceKind.OBSERVATION
    ]
    states = [record for record in records if record.kind == SourceKind.STATE]
    if len(actions) == 1 and len(observations) == 1 and (not states):
        payload: Any = {
            "action": _source_value(actions[0]),
            "observation": _source_value(observations[0]),
        }
    else:
        payload = {
            "sources": [
                {"kind": record.kind.value, "content": _source_value(record)}
                for record in records
            ]
        }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _source_value(record: SourceRecord) -> Any:
    if isinstance(record.payload, str):
        return record.payload
    return json.loads(source_payload_text(record))


def _supplement_retrieved(graph: MemoryGraph, retrieved, candidate_ids: Sequence[str]):
    selected = list(retrieved.memory_ids)
    for memory_id in candidate_ids:
        if memory_id in graph.units and memory_id not in selected:
            selected.append(memory_id)
    selected_set = set(selected)
    relation_ids = tuple(
        (
            relation.id
            for relation in graph.relations.values()
            if relation.source in selected_set and relation.target in selected_set
        )
    )
    return replace(
        retrieved,
        memory_ids=tuple(selected),
        relation_ids=relation_ids,
        closure_groups=tuple(((memory_id,) for memory_id in selected)),
    )


def _jsonable(value: Any) -> Any:
    if hasattr(value, "value") and type(getattr(value, "value")).__name__ in {
        "str",
        "int",
        "float",
    }:
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list | set | frozenset):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _jsonable(getattr(value, name)) for name in value.__dataclass_fields__
        }
    return value
