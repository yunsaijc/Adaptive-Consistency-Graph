from __future__ import annotations

import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

from .evidence import EvidenceRecord, extract_complete_evidence_records
from .schema import (
    AddMemory,
    AppliedDelta,
    GraphDelta,
    MemoryUnit,
    RejectedOperation,
    SourceKind,
    SourceRecord,
)
from .store import MemoryGraph, source_payload_text

BUILDER_SYSTEM_PROMPT = "You select compact, source-grounded memory for a general-purpose agent.\n\nThe graph is append-only memory, not a task controller. The persistent task is a read-only relevance reference. New\nsource records are the only extraction input. An action record provides context about what the agent executed but does\nnot establish facts about the external world. An observation or state record reports what the environment returned.\n\nPreserve only new information that may affect a later decision. Do not copy the task, store plans or reasoning, narrate\nprogress, or turn an executed action into a factual memory. Existing memory candidates are read-only context for avoiding\nrepetition. Never rewrite or delete them.\n\nUse only the add operation. The runtime projects each new observation or state into complete evidence records and immutable\nunits. Each add operation selects one record and the smallest set of its units that preserves enough identity and context\nto remain understandable later. Never quote, copy, paraphrase, combine, or infer memory text: the runtime renders selected\nunits exactly. Records without evidence units are context only and cannot be selected. Return no operation when no relevant\nevidence exists. The runtime builds retrieval relations deterministically from embeddings; do not propose, classify, or\ninfer relations.\n\nThe runtime attaches immutable source records. Do not output source ids, memory text, quotes, confidence scores, relations,\nor weights. Refer to a memory added in this response by its temp_id. Do not infer task completion, plan actions, select\ntools, assign status, or use a domain-specific schema. Return only the requested JSON.\n"
BUILDER_OUTPUT_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"const": "add"},
                    "temp_id": {"type": "string"},
                    "record_id": {"type": "string"},
                    "unit_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                },
                "required": ["op", "temp_id", "record_id", "unit_ids"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["operations"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class BuildContext:
    task: SourceRecord
    event_sources: tuple[SourceRecord, ...]
    related_memories: tuple[MemoryUnit, ...] = ()


@dataclass(frozen=True)
class BuilderProposal:
    delta: GraphDelta
    parse_errors: tuple[str, ...] = ()
    raw_output: Mapping[str, Any] | str | None = None


class StructuredGenerationClient(Protocol):
    def complete_json(
        self, *, system_prompt: str, user_prompt: str, schema: Mapping[str, Any]
    ) -> Mapping[str, Any] | str:
        raise NotImplementedError


class MemoryBuilder(Protocol):
    def propose(self, context: BuildContext) -> BuilderProposal:
        raise NotImplementedError


@dataclass(frozen=True)
class CompleteEvidenceWrite:
    applied: AppliedDelta
    records_by_source: Mapping[str, tuple[EvidenceRecord, ...]]
    memory_ids_by_source: Mapping[str, tuple[str, ...]]


def write_complete_evidence(
    graph: MemoryGraph,
    event_sources: Sequence[SourceRecord],
    *,
    evidence_extractor: Callable[..., tuple[EvidenceRecord, ...]] | None = None,
) -> CompleteEvidenceWrite:
    extractor = evidence_extractor or extract_complete_evidence_records
    added: list[str] = []
    reused: list[str] = []
    records_by_source: dict[str, tuple[EvidenceRecord, ...]] = {}
    memory_ids_by_source: dict[str, tuple[str, ...]] = {}
    for source in event_sources:
        if source.kind not in {SourceKind.OBSERVATION, SourceKind.STATE}:
            continue
        records = extractor(source_payload_text(source), id_prefix=source.id)
        records_by_source[source.id] = records
        source_memory_ids: list[str] = []
        for record in records:
            content = "\n".join((unit.text for unit in record.units))
            if not content.strip():
                continue
            existing = graph.find_memory_by_content(content)
            memory = graph.add_memory(content=content, source_refs=(source.id,))
            source_memory_ids.append(memory.id)
            if existing is None:
                added.append(memory.id)
            else:
                reused.append(memory.id)
        memory_ids_by_source[source.id] = tuple(source_memory_ids)
    return CompleteEvidenceWrite(
        applied=AppliedDelta(
            added_memory_ids=tuple(dict.fromkeys(added)),
            reused_memory_ids=tuple(dict.fromkeys(reused)),
        ),
        records_by_source=MappingProxyType(records_by_source),
        memory_ids_by_source=MappingProxyType(memory_ids_by_source),
    )


class LLMMemoryBuilder:
    def __init__(self, client: StructuredGenerationClient) -> None:
        self.client = client
        self.calls = 0

    def propose(self, context: BuildContext) -> BuilderProposal:
        self.calls += 1
        payload = self.client.complete_json(
            system_prompt=BUILDER_SYSTEM_PROMPT,
            user_prompt=render_builder_input(context),
            schema=BUILDER_OUTPUT_SCHEMA,
        )
        return parse_builder_output(payload, context=context)


class StaticMemoryBuilder:
    def __init__(self, proposal: BuilderProposal | GraphDelta) -> None:
        self.proposal = (
            proposal
            if isinstance(proposal, BuilderProposal)
            else BuilderProposal(proposal)
        )
        self.calls = 0
        self.contexts: list[BuildContext] = []

    def propose(self, context: BuildContext) -> BuilderProposal:
        self.calls += 1
        self.contexts.append(context)
        return self.proposal


def render_builder_input(context: BuildContext) -> str:
    evidence_records = _evidence_records_by_source(context)
    payload = {
        "persistent_task_reference": {
            "write_policy": "read_only_relevance_reference_do_not_write",
            "record": _source_to_dict(context.task),
        },
        "new_sources_to_reconcile": [
            _source_to_builder_dict(
                record, evidence_records=evidence_records.get(record.id, ())
            )
            for record in context.event_sources
        ],
        "existing_memory_candidates": [
            {
                "id": unit.id,
                "content": unit.content,
                "source_refs": list(unit.source_refs),
            }
            for unit in context.related_memories
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def parse_builder_output(
    payload: Mapping[str, Any] | str, *, context: BuildContext
) -> BuilderProposal:
    raw_output: Mapping[str, Any] | str = (
        payload if isinstance(payload, str) else dict(payload)
    )
    if isinstance(payload, str):
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as error:
            return BuilderProposal(
                GraphDelta(), (f"Invalid JSON: {error.msg}",), raw_output
            )
    else:
        decoded = payload
    if not isinstance(decoded, Mapping):
        return BuilderProposal(
            GraphDelta(), ("Builder output must be a JSON object",), raw_output
        )
    raw_operations = decoded.get("operations")
    if not isinstance(raw_operations, list):
        return BuilderProposal(
            GraphDelta(),
            ("Builder output must contain an operations array",),
            raw_output,
        )
    operations = []
    errors: list[str] = []
    evidence_records = _evidence_records(context)
    for index, raw in enumerate(raw_operations):
        try:
            operations.append(_parse_operation(raw, evidence_records=evidence_records))
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"operation[{index}]: {error}")
    return BuilderProposal(GraphDelta(tuple(operations)), tuple(errors), raw_output)


def apply_delta(
    graph: MemoryGraph, delta: GraphDelta, *, event_source_ids: Sequence[str]
) -> AppliedDelta:
    event_refs = graph.sources.require(event_source_ids)
    if any(
        (
            graph.sources.get(source_id).kind == SourceKind.TASK
            for source_id in event_refs
        )
    ):
        raise ValueError("The persistent task cannot be used as graph-write provenance")
    externally_grounded = any(
        (
            graph.sources.get(source_id).kind
            in {SourceKind.OBSERVATION, SourceKind.STATE}
            for source_id in event_refs
        )
    )
    rejected: list[RejectedOperation] = []
    temp_id_map: dict[str, str] = {}
    added: list[str] = []
    reused: list[str] = []
    for index, operation in enumerate(delta.operations):
        try:
            if operation.temp_id in temp_id_map:
                raise ValueError(f"Duplicate temp id: {operation.temp_id}")
            if not externally_grounded:
                raise ValueError("Memory writes require an observation or state source")
            existing = graph.find_memory_by_content(operation.content)
            unit = graph.add_memory(content=operation.content, source_refs=event_refs)
            temp_id_map[operation.temp_id] = unit.id
            if existing is None:
                added.append(unit.id)
            else:
                reused.append(unit.id)
        except (KeyError, TypeError, ValueError) as error:
            rejected.append(RejectedOperation(index, str(error)))
    rejected.sort(key=lambda item: item.operation_index)
    return AppliedDelta(
        added_memory_ids=tuple(dict.fromkeys(added)),
        reused_memory_ids=tuple(dict.fromkeys(reused)),
        temp_id_map=MappingProxyType(dict(temp_id_map)),
        rejected=tuple(rejected),
    )


def _parse_operation(raw: Any, *, evidence_records: Mapping[str, EvidenceRecord]):
    if not isinstance(raw, Mapping):
        raise TypeError("operation must be an object")
    op = _required_text(raw, "op")
    if op == "add":
        _require_exact_fields(raw, {"op", "temp_id", "record_id", "unit_ids"})
        record_id = _required_text(raw, "record_id")
        if record_id not in evidence_records:
            raise ValueError(f"unknown or non-selectable record_id: {record_id}")
        raw_unit_ids = raw.get("unit_ids")
        if not isinstance(raw_unit_ids, list) or not raw_unit_ids:
            raise ValueError("unit_ids must be a non-empty array")
        if any(
            (
                not isinstance(unit_id, str) or not unit_id.strip()
                for unit_id in raw_unit_ids
            )
        ):
            raise ValueError("unit_ids must contain non-empty strings")
        selected_ids = set(raw_unit_ids)
        record = evidence_records[record_id]
        known_ids = {unit.id for unit in record.units}
        unknown_ids = sorted(selected_ids - known_ids)
        if unknown_ids:
            raise ValueError(f"unknown unit_ids for {record_id}: {unknown_ids}")
        content = "\n".join(
            (unit.text for unit in record.units if unit.id in selected_ids)
        )
        return AddMemory(temp_id=_required_text(raw, "temp_id"), content=content)
    raise ValueError(f"unknown operation: {op}")


def _require_exact_fields(raw: Mapping[str, Any], expected: set[str]) -> None:
    unexpected = sorted(set(raw) - expected)
    if unexpected:
        raise ValueError(f"unexpected fields: {unexpected}")


def _required_text(raw: Mapping[str, Any], key: str) -> str:
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _source_to_dict(record: SourceRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "kind": record.kind.value,
        "content": source_payload_text(record),
        "sequence": record.sequence,
        "timestamp": record.timestamp,
    }


def _source_to_builder_dict(
    record: SourceRecord, *, evidence_records: Sequence[EvidenceRecord]
) -> dict[str, Any]:
    item = _source_to_dict(record)
    if record.kind == SourceKind.ACTION:
        item["content"] = builder_source_text(record)
        item["evidence_role"] = "executed_agent_action_not_external_fact"
    elif record.kind == SourceKind.OBSERVATION:
        item["evidence_role"] = "external_observation"
    else:
        item["evidence_role"] = "recorded_state"
    if record.kind in {SourceKind.OBSERVATION, SourceKind.STATE}:
        item.pop("content", None)
        item["complete_evidence_records"] = [
            {
                "record_id": evidence_record.id,
                "units": [
                    {"unit_id": unit.id, "text": unit.text}
                    for unit in evidence_record.units
                ],
            }
            for evidence_record in evidence_records
        ]
    return item


def _evidence_records_by_source(
    context: BuildContext,
) -> dict[str, tuple[EvidenceRecord, ...]]:
    projected: dict[str, tuple[EvidenceRecord, ...]] = {}
    evidence_index = 0
    for record in context.event_sources:
        if record.kind not in {SourceKind.OBSERVATION, SourceKind.STATE}:
            continue
        evidence_index += 1
        projected[record.id] = extract_complete_evidence_records(
            source_payload_text(record), id_prefix=f"evidence_{evidence_index}"
        )
    return projected


def _evidence_records(context: BuildContext) -> dict[str, EvidenceRecord]:
    return {
        record.id: record
        for records in _evidence_records_by_source(context).values()
        for record in records
    }


def builder_source_text(record: SourceRecord) -> str:
    text = source_payload_text(record)
    if record.kind != SourceKind.ACTION:
        return text
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(decoded, Mapping) or "action" not in decoded:
        return text
    executed = {"action": decoded["action"], "arguments": decoded.get("arguments", {})}
    return json.dumps(executed, ensure_ascii=False, sort_keys=True)
