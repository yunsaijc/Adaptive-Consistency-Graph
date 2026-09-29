from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from .builder import StructuredGenerationClient
from .evidence import extract_complete_evidence_records
from .organization import EventOrganization
from .relations import cosine_similarity
from .schema import SourceKind, SourceRecord
from .store import source_payload_text

CONCEPT_SYSTEM_PROMPT = "You index source-grounded concepts for a general-purpose agent.\n\nA concept only reconnects source records about the same concrete area of work. It is not a fact, summary, task branch,\nplan, state, dependency, or causal relation. A record may belong to zero, one, or several concepts.\n\nFor each current record, you may reference provided existing concept ids and create new anchors by selecting exact unit ids\nfrom that record. Use the smallest source-grounded identity that can recognize the work later. Do not select generic tool\nnames, formatting fields, or common metadata alone. Do not invent text, names, explanations, types, confidence, weights,\nstatus, or relations. Use only ids from the input and return only the requested JSON."
CONCEPT_OUTPUT_CONTRACT = 'Output contract:\n- Return exactly one JSON object with these top-level keys: "records", "action_origins", and "user_source_decisions".\n- "records" must be an array. Each item must contain exactly: "record_id", "existing_concept_ids", "new_anchors".\n- "action_origins" must be an array of source-grounded action assignments.\n- "user_source_decisions" must be an array of complete decisions for pending later-user sources, or an empty array when there are none.\n- "record_id" must be copied from current_source_records[*].record_id.\n- "existing_concept_ids" may only contain ids from existing_concept_candidates[*].concept_id.\n- "new_anchors" is an array of arrays of unit ids copied from the same record\'s units[*].unit_id.\n- Do not output concept_id, concept_name, name, type, source_record_ids, anchor_unit_ids, anchors, concepts, summary, status, relation, or explanation.\n\nValid shape:\n{"records":[{"record_id":"<record_id>","existing_concept_ids":[],"new_anchors":[["<unit_id>"]]}],"action_origins":[],"user_source_decisions":[]}\n'


@dataclass(frozen=True)
class ConceptUnit:
    id: str
    text: str


@dataclass(frozen=True)
class ConceptRecord:
    id: str
    source_id: str
    source_kind: str
    units: tuple[ConceptUnit, ...]
    payload: Any = None


@dataclass(frozen=True)
class ConceptAssignment:
    record_id: str
    existing_concept_ids: tuple[str, ...]
    new_anchor_texts: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class ConceptProposal:
    assignments: tuple[ConceptAssignment, ...] = ()
    parse_errors: tuple[str, ...] = ()
    raw_output: Mapping[str, Any] | str | None = None


@dataclass(frozen=True)
class ConceptContext:
    task: SourceRecord
    records: tuple[ConceptRecord, ...]
    candidate_ids: tuple[str, ...]
    organization: EventOrganization


class ConceptIndexer(Protocol):
    def propose(self, context: ConceptContext) -> ConceptProposal:
        raise NotImplementedError


class LLMConceptIndexer:
    def __init__(self, client: StructuredGenerationClient) -> None:
        self.client = client
        self.calls = 0

    def propose(self, context: ConceptContext) -> ConceptProposal:
        if not context.records:
            return ConceptProposal()
        self.calls += 1
        payload = self.client.complete_json(
            system_prompt=concept_system_prompt(),
            user_prompt=render_concept_input(context),
            schema=concept_output_schema(context),
        )
        return parse_concept_output(payload, context=context)


class StaticConceptIndexer:
    def __init__(self, proposal: ConceptProposal) -> None:
        self.proposal = proposal
        self.calls = 0
        self.contexts: list[ConceptContext] = []

    def propose(self, context: ConceptContext) -> ConceptProposal:
        self.calls += 1
        self.contexts.append(context)
        return self.proposal


def concept_records(
    sources: Sequence[SourceRecord],
    *,
    evidence_extractor: Callable[..., Any] | None = None,
) -> tuple[ConceptRecord, ...]:
    records: list[ConceptRecord] = []
    extractor = evidence_extractor or extract_complete_evidence_records
    for source in sources:
        projected = extractor(source_payload_text(source), id_prefix=source.id)
        for record in projected:
            units = tuple(
                (
                    ConceptUnit(unit.id, unit.text)
                    for unit in record.units
                    if unit.text.strip()
                )
            )
            if not units:
                continue
            records.append(
                ConceptRecord(
                    id=record.id,
                    source_id=source.id,
                    source_kind=source.kind.value,
                    units=units,
                    payload=source.payload,
                )
            )
    return tuple(records)


def concept_candidates(
    organization: EventOrganization,
    *,
    records: Sequence[ConceptRecord],
    query_embedding: Sequence[float],
    top_k: int,
) -> tuple[str, ...]:
    if top_k <= 0:
        raise ValueError("Concept candidate limit must be positive")
    if not organization.concepts:
        return ()
    exact_texts = {unit.text for record in records for unit in record.units}
    exact = {
        concept.id
        for concept in organization.concepts.values()
        if set(concept.anchor_texts).issubset(exact_texts)
    }
    ranked = sorted(
        (
            concept
            for concept in organization.concepts.values()
            if concept.embedding is not None
        ),
        key=lambda concept: (
            -cosine_similarity(query_embedding, concept.embedding or ()),
            concept.id,
        ),
    )
    return tuple(
        dict.fromkeys((*sorted(exact), *(concept.id for concept in ranked[:top_k])))
    )


def apply_concept_proposal(
    organization: EventOrganization,
    proposal: ConceptProposal,
    *,
    event_id: str,
    records: Sequence[ConceptRecord] | None = None,
    evidence_ids_by_record: Mapping[str, Sequence[str]] | None = None,
) -> tuple[str, ...]:
    active: list[str] = []
    record_kinds = {record.id: record.source_kind for record in records or ()}
    record_evidence = evidence_ids_by_record or {}
    for assignment in proposal.assignments:
        is_active = (
            not record_kinds
            or record_kinds.get(assignment.record_id) == SourceKind.ACTION.value
        )
        for concept_id in assignment.existing_concept_ids:
            organization.attach_concept(
                concept_id,
                event_id=event_id,
                active=is_active,
                evidence_ids=record_evidence.get(assignment.record_id, ()),
            )
            active.append(concept_id)
        for anchor_texts in assignment.new_anchor_texts:
            active.append(
                organization.create_or_reuse_concept(
                    anchor_texts,
                    event_id=event_id,
                    active=is_active,
                    evidence_ids=record_evidence.get(assignment.record_id, ()),
                )
            )
    return tuple(dict.fromkeys(active))


def apply_exact_concept_memberships(
    organization: EventOrganization,
    *,
    event_id: str,
    records: Sequence[ConceptRecord],
    evidence_ids_by_record: Mapping[str, Sequence[str]] | None = None,
) -> tuple[str, ...]:
    attached: list[str] = []
    record_evidence = evidence_ids_by_record or {}
    for record in records:
        record_texts = {unit.text for unit in record.units}
        for concept in tuple(organization.concepts.values()):
            if set(concept.anchor_texts).issubset(record_texts):
                organization.attach_concept(
                    concept.id,
                    event_id=event_id,
                    active=record.source_kind == SourceKind.ACTION.value,
                    evidence_ids=record_evidence.get(record.id, ()),
                )
                attached.append(concept.id)
    return tuple(dict.fromkeys(attached))


def concepts_missing_embeddings(
    organization: EventOrganization, concept_ids: Sequence[str]
) -> tuple[str, ...]:
    return tuple(
        (
            concept_id
            for concept_id in dict.fromkeys(concept_ids)
            if organization.concepts[concept_id].embedding is None
        )
    )


def render_concept_input(context: ConceptContext) -> str:
    payload = {
        "persistent_task_reference": source_payload_text(context.task),
        "existing_concept_candidates": [
            {
                "concept_id": concept_id,
                "source_anchor_units": list(
                    context.organization.concepts[concept_id].anchor_texts
                ),
            }
            for concept_id in context.candidate_ids
        ],
        "current_source_records": [
            {
                "record_id": record.id,
                "source_kind": record.source_kind,
                "units": [
                    {"unit_id": unit.id, "text": unit.text} for unit in record.units
                ],
            }
            for record in context.records
        ],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def concept_system_prompt() -> str:
    return f"{CONCEPT_SYSTEM_PROMPT}\n\n{CONCEPT_OUTPUT_CONTRACT}"


def concept_output_schema(context: ConceptContext) -> Mapping[str, Any]:
    record_ids = [record.id for record in context.records]
    unit_ids = [unit.id for record in context.records for unit in record.units]
    existing_items: dict[str, Any] = {"type": "string"}
    existing_array: dict[str, Any] = {"type": "array", "items": existing_items}
    if context.candidate_ids:
        existing_items["enum"] = list(context.candidate_ids)
    else:
        existing_array["maxItems"] = 0
    records = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "record_id": {"type": "string", "enum": record_ids},
                "existing_concept_ids": existing_array,
                "new_anchors": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "string", "enum": unit_ids},
                        "minItems": 1,
                    },
                },
            },
            "required": ["record_id", "existing_concept_ids", "new_anchors"],
            "additionalProperties": False,
        },
    }
    action_origins = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "action_source_id": {"type": "string"},
                "tool_call_id": {"type": "string"},
                "requirement_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["action_source_id", "tool_call_id", "requirement_ids"],
            "additionalProperties": False,
        },
    }
    user_source_decisions = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "source_id": {"type": "string"},
                "segments": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "exact_span": {"type": "string", "minLength": 1},
                            "decision": {
                                "type": "string",
                                "enum": ["new_requirement", "continuation_evidence"],
                            },
                            "continuation_requirement_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "exact_span",
                            "decision",
                            "continuation_requirement_ids",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["source_id", "segments"],
            "additionalProperties": False,
        },
    }
    return {
        "type": "object",
        "properties": {
            "records": records,
            "action_origins": action_origins,
            "user_source_decisions": user_source_decisions,
        },
        "required": ["records", "action_origins", "user_source_decisions"],
        "additionalProperties": False,
    }


def parse_concept_output(
    payload: Mapping[str, Any] | str, *, context: ConceptContext
) -> ConceptProposal:
    raw_output: Mapping[str, Any] | str = (
        payload if isinstance(payload, str) else dict(payload)
    )
    if isinstance(payload, str):
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as error:
            return ConceptProposal(
                parse_errors=(f"Invalid JSON: {error.msg}",), raw_output=raw_output
            )
    else:
        decoded = payload
    required_fields = {"records", "action_origins", "user_source_decisions"}
    if not isinstance(decoded, Mapping) or set(decoded) != required_fields:
        return ConceptProposal(
            parse_errors=(
                "Concept output must contain records, action_origins, and user_source_decisions arrays",
            ),
            raw_output=raw_output,
        )
    raw_records = decoded.get("records")
    if not isinstance(raw_records, list):
        return ConceptProposal(
            parse_errors=("Concept output records must be an array",),
            raw_output=raw_output,
        )
    records = {record.id: record for record in context.records}
    candidates = set(context.candidate_ids)
    assignments: list[ConceptAssignment] = []
    errors: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_records):
        try:
            assignment = _parse_assignment(raw, records=records, candidates=candidates)
            if assignment.record_id in seen:
                raise ValueError("duplicate record_id")
            seen.add(assignment.record_id)
            assignments.append(assignment)
        except (TypeError, ValueError) as error:
            errors.append(f"records[{index}]: {error}")
    return ConceptProposal(tuple(assignments), tuple(errors), raw_output)


def _parse_assignment(
    raw: Any, *, records: Mapping[str, ConceptRecord], candidates: set[str]
) -> ConceptAssignment:
    if not isinstance(raw, Mapping):
        raise TypeError("record must be an object")
    if set(raw) != {"record_id", "existing_concept_ids", "new_anchors"}:
        raise ValueError("unexpected fields")
    record_id = raw.get("record_id")
    if record_id not in records:
        raise ValueError("unknown record_id")
    existing = raw.get("existing_concept_ids")
    if not isinstance(existing, list) or any(
        (item not in candidates for item in existing)
    ):
        raise ValueError("unknown existing concept")
    anchors = raw.get("new_anchors")
    if not isinstance(anchors, list):
        raise ValueError("new_anchors must be an array")
    unit_text = {unit.id: unit.text for unit in records[str(record_id)].units}
    parsed_anchors: list[tuple[str, ...]] = []
    for anchor in anchors:
        if (
            not isinstance(anchor, list)
            or not anchor
            or len(anchor) != len(set(anchor))
            or any((unit_id not in unit_text for unit_id in anchor))
        ):
            raise ValueError("anchor must contain unique units from its own record")
        parsed_anchors.append(tuple((unit_text[unit_id] for unit_id in anchor)))
    return ConceptAssignment(
        record_id=str(record_id),
        existing_concept_ids=tuple(dict.fromkeys((str(item) for item in existing))),
        new_anchor_texts=tuple(parsed_anchors),
    )
