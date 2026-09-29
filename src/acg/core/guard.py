from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from .concepts import (
    ConceptContext,
    ConceptIndexer,
    ConceptProposal,
    parse_concept_output,
)


@dataclass(frozen=True)
class BracketRepair:
    payload: str
    inserted_brackets: int
    positions: tuple[int, ...]


DEFAULT_TRUNCATION_RETRY_MAX_TOKENS = 4096
OUTCOME_SYNTAX_SAFE_REPAIRED = "syntax_safe_repaired"
OUTCOME_TRUNCATION_RETRY = "truncation_retry_applied_or_failed"
OUTCOME_SEMANTIC_REJECTED = "semantic_rejected"
OUTCOME_STRUCTURAL_UNREPAIRABLE = "structural_unrepairable"
ACTION_ORIGINS_SCHEMA_VERSION = "action_origins.v1"
USER_SOURCE_DECISIONS_SCHEMA_VERSION = "user_source_decisions.v1"


def _recover_action_origins_payload(
    proposal: ConceptProposal, context: ConceptContext
) -> ConceptProposal:
    raw = proposal.raw_output
    decoded: Any = raw
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return proposal
    if not isinstance(decoded, Mapping):
        return proposal
    reparsed = parse_concept_output(decoded, context=context)
    return reparsed if not reparsed.parse_errors else proposal


def _parse_core_output(
    payload: Mapping[str, Any] | str, *, context: ConceptContext
) -> ConceptProposal:
    return parse_concept_output(payload, context=context)


def _action_occurrence_catalog(
    context: ConceptContext, *, source_payload_provider: Any = None
) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for record in context.records:
        if str(record.source_kind) != "action":
            continue
        payload: Any = getattr(record, "payload", None)
        if payload is None and callable(source_payload_provider):
            try:
                payload = source_payload_provider(str(record.source_id))
            except Exception:
                payload = None
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (TypeError, json.JSONDecodeError):
                payload = None
        calls = payload.get("tool_calls") if isinstance(payload, Mapping) else ()
        if not isinstance(calls, list | tuple):
            if isinstance(payload, Mapping) and any(
                (
                    key in payload
                    for key in ("function", "tool", "name", "arguments", "args")
                )
            ):
                calls = (payload,)
            else:
                calls = ()
        for call in calls or ():
            if not isinstance(call, Mapping):
                continue
            function = (
                call.get("function")
                if isinstance(call.get("function"), Mapping)
                else {}
            )
            arguments: Any = function.get("arguments", call.get("arguments", {}))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (TypeError, json.JSONDecodeError):
                    pass
            catalog.append(
                {
                    "action_source_id": str(record.source_id),
                    "tool_call_id": str(call.get("id", call.get("tool_call_id", ""))),
                    "tool_name": str(function.get("name", call.get("name", ""))),
                    "arguments": _jsonable_origin_value(arguments),
                }
            )
    return catalog


def _jsonable_origin_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable_origin_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_origin_value(item) for item in value]
    return value


def _validate_exact_span_coverage(source_text: str, spans: Any) -> str | None:
    if not isinstance(spans, list) or not spans:
        return "exact_spans_do_not_cover_source"
    intervals: list[tuple[int, int]] = []
    for span in spans:
        if not isinstance(span, str) or not span:
            return "exact_span_not_string"
        start = source_text.find(span)
        if start < 0:
            return "exact_span_not_grounded"
        end = start + len(span)
        if any((not (end <= left or start >= right) for left, right in intervals)):
            return "exact_span_duplicate_or_overlapping"
        intervals.append((start, end))
    intervals.sort()
    cursor = 0
    for start, end in intervals:
        if any((not char.isspace() for char in source_text[cursor:start])):
            return "exact_spans_leave_source_uncovered"
        cursor = end
    if any((not char.isspace() for char in source_text[cursor:])):
        return "exact_spans_leave_source_uncovered"
    return None


def _is_semantic_rejection(parse_errors: tuple[str, ...]) -> bool:
    markers = (
        "unexpected fields",
        "unknown record_id",
        "unknown existing concept",
        "duplicate record_id",
        "anchor must contain unique units from its own record",
    )
    return any((marker in error for error in parse_errors for marker in markers))


def _unrepaired_outcome(parse_errors: tuple[str, ...]) -> str:
    return (
        OUTCOME_SEMANTIC_REJECTED
        if _is_semantic_rejection(parse_errors)
        else OUTCOME_STRUCTURAL_UNREPAIRABLE
    )


def repair_missing_new_anchors_brackets(
    payload: str, *, context: ConceptContext | None = None
) -> BracketRepair | None:
    if not isinstance(payload, str) or not payload.strip():
        return None
    try:
        json.loads(payload)
    except json.JSONDecodeError:
        pass
    else:
        return None
    key_spans = _string_key_spans(payload, key="new_anchors")
    if key_spans is None:
        return None
    insertions: list[int] = []
    for _start, key_end in key_spans:
        value_start = _skip_space(payload, key_end)
        if value_start >= len(payload) or payload[value_start] != ":":
            continue
        value_start = _skip_space(payload, value_start + 1)
        if value_start >= len(payload) or payload[value_start] != "[":
            continue
        insertion = _scan_anchor_array(payload, value_start)
        if insertion is _AMBIGUOUS:
            return None
        if insertion is not None:
            insertions.append(insertion)
    if not insertions or len(set(insertions)) != len(insertions):
        return None
    repaired = payload
    for position in reversed(insertions):
        repaired = repaired[:position] + "]" + repaired[position:]
    try:
        json.loads(repaired)
    except json.JSONDecodeError:
        return None
    if context is not None:
        if _parse_core_output(repaired, context=context).parse_errors:
            return None
    return BracketRepair(
        payload=repaired, inserted_brackets=len(insertions), positions=tuple(insertions)
    )


class ConceptIndexerGuard:
    def __init__(
        self,
        delegate: ConceptIndexer,
        *,
        truncation_retry_max_tokens: int = DEFAULT_TRUNCATION_RETRY_MAX_TOKENS,
    ) -> None:
        if truncation_retry_max_tokens <= 0:
            raise ValueError("ACG truncation retry max_tokens must be positive")
        self.delegate = delegate
        self.truncation_retry_max_tokens = truncation_retry_max_tokens
        self.repair_attempts = 0
        self.repairs_applied = 0
        self.truncation_retry_attempts = 0
        self.truncation_retries_applied = 0
        self.last_proposal: ConceptProposal | None = None
        self._origin_requirements_provider: Any = lambda: ()
        self._source_payload_provider: Any = None
        self._later_user_sources_provider: Any = lambda: ()
        self._nearest_assistant_context_provider: Any = lambda: None
        self._dynamic_requirements_handler: Any = None
        self.last_dynamic_requirements: dict[str, Any] = {
            "status": "absent",
            "decision_status": "absent",
            "decisions": [],
            "pending_source_ids": [],
            "appended": [],
            "issues": [],
        }

    def set_origin_requirements_provider(self, provider: Any) -> None:
        self._origin_requirements_provider = provider

    def set_source_payload_provider(self, provider: Any) -> None:
        self._source_payload_provider = provider

    def set_later_user_sources_provider(self, provider: Any) -> None:
        self._later_user_sources_provider = provider

    def set_nearest_assistant_context_provider(self, provider: Any) -> None:
        self._nearest_assistant_context_provider = provider

    def set_dynamic_requirements_handler(self, handler: Any) -> None:
        self._dynamic_requirements_handler = handler

    def _origin_context(self, context: ConceptContext) -> str:
        requirements = tuple(self._origin_requirements_provider() or ())
        pending_user_sources = []
        try:
            candidates = tuple(self._later_user_sources_provider() or ())
        except Exception:
            candidates = ()
        for item in candidates:
            if isinstance(item, Mapping):
                source_id = item.get("source_id")
                text = item.get("text", item.get("exact_text", ""))
                sequence = item.get("sequence")
                preceding_source_id = item.get("preceding_assistant_source_id")
                preceding_text = item.get("preceding_assistant_text")
            else:
                source_id = getattr(item, "source_id", None)
                text = getattr(item, "text", "")
                sequence = getattr(item, "sequence", None)
                status = getattr(item, "status", "pending")
                preceding_source_id = getattr(
                    item, "preceding_assistant_source_id", None
                )
                preceding_text = getattr(item, "preceding_assistant_text", None)
            if source_id in (None, "") or not isinstance(text, str):
                continue
            if isinstance(item, Mapping):
                status = item.get("status", "pending")
            if str(status) != "pending":
                continue
            entry = {"source_id": str(source_id), "text": text}
            if sequence is not None:
                entry["sequence"] = sequence
            if preceding_source_id is not None or preceding_text is not None:
                entry["preceding_assistant_context"] = {
                    "source_id": preceding_source_id,
                    "text": preceding_text,
                }
            pending_user_sources.append(entry)
        frozen_assistant = [
            entry["preceding_assistant_context"]
            for entry in pending_user_sources
            if entry.get("preceding_assistant_context")
        ]
        try:
            nearest_assistant = (
                frozen_assistant
                if pending_user_sources
                else self._nearest_assistant_context_provider()
            )
        except Exception:
            nearest_assistant = None
        if nearest_assistant is None:
            action_catalog = _action_occurrence_catalog(
                context, source_payload_provider=self._source_payload_provider
            )
            nearest_assistant = action_catalog[-1:] if action_catalog else []
        elif isinstance(nearest_assistant, Mapping):
            nearest_assistant = [dict(nearest_assistant)]
        elif not isinstance(nearest_assistant, (list, tuple)):
            nearest_assistant = []
        catalog = {
            "schema_version": USER_SOURCE_DECISIONS_SCHEMA_VERSION,
            "requirements": [
                {
                    "id": str(item.id),
                    "source_id": str(item.source_id),
                    "exact_text": str(item.exact_text),
                }
                for item in requirements
            ],
            "nearest_assistant_context": nearest_assistant,
            "pending_user_sources": pending_user_sources,
        }
        return (
            "ACG structured origin contract. Keep the core `records` array unchanged. Add an `action_origins` array with exactly one row for every listed native action call, copying action_source_id and tool_call_id exactly. requirement_ids may contain only the catalog IDs. For each action, use the listed tool_name and structured arguments, including an explicit query argument when present, to make a source-grounded match against the requirement catalog. When the tool and arguments clearly support one or more requirements, return those requirement IDs; use [] only for genuinely cross-requirement actions or when no clear match is supported. Do not use embeddings, action order, or generic scalar values alone. For every pending user source, add exactly one `user_source_decisions` row. Use decision `new_requirement` with segments that cover its source, or `continuation_evidence` segments plus existing continuation_requirement_ids. Use an empty array only when there are no pending user sources. A source is continuation evidence by default when it answers, confirms, supplies lookup/payment details for, or otherwise responds to the preceding assistant request about an existing requirement. Use `new_requirement` only for an explicit independent outcome or object not already covered. If one user source mixes both, split it into continuation and new segments; do not turn the whole source into a new requirement.\n"
            + json.dumps(catalog, ensure_ascii=False, sort_keys=True)
        )

    def apply_dynamic_requirements(self, payload: Any) -> dict[str, Any]:
        decoded: Any = (
            payload.raw_output if isinstance(payload, ConceptProposal) else payload
        )
        if isinstance(decoded, str):
            try:
                decoded = json.loads(decoded)
            except (TypeError, json.JSONDecodeError):
                decoded = None
        if isinstance(decoded, Mapping) and "user_source_decisions" in decoded:
            return self.apply_user_source_decisions(decoded)
        try:
            pending = tuple(self._later_user_sources_provider() or ())
        except Exception:
            pending = ()
        pending_ids = [
            str(item.get("source_id"))
            if isinstance(item, Mapping)
            else str(getattr(item, "source_id", ""))
            for item in pending
            if (
                item.get("status", "pending")
                if isinstance(item, Mapping)
                else getattr(item, "status", "pending")
            )
            == "pending"
        ]
        result: dict[str, Any] = {
            "status": "absent",
            "decision_status": "absent",
            "decisions": [],
            "pending_source_ids": pending_ids,
            "appended": [],
            "issues": [],
        }
        self.last_dynamic_requirements = result
        return result

    def apply_user_source_decisions(self, payload: Any) -> dict[str, Any]:
        decisions = (
            payload.get("user_source_decisions")
            if isinstance(payload, Mapping)
            else None
        )
        try:
            catalog = tuple(self._later_user_sources_provider() or ())
        except Exception:
            catalog = ()
        pending = tuple(
            (
                item
                for item in catalog
                if (
                    item.get("status", "pending")
                    if isinstance(item, Mapping)
                    else getattr(item, "status", "pending")
                )
                == "pending"
            )
        )
        pending_ids = tuple(
            (
                str(item.get("source_id"))
                if isinstance(item, Mapping)
                else str(getattr(item, "source_id", ""))
                for item in pending
            )
        )
        result: dict[str, Any] = {
            "status": "invalid",
            "decision_status": "invalid",
            "decisions": list(decisions) if isinstance(decisions, list) else [],
            "pending_source_ids": list(pending_ids),
            "appended": [],
            "issues": [],
        }
        if not isinstance(decisions, list):
            result["issues"] = ["user_source_decisions_not_array"]
            self.last_dynamic_requirements = result
            return result
        if not pending and (not decisions):
            result.update({"status": "not_needed", "decision_status": "not_needed"})
            self.last_dynamic_requirements = result
            return result
        if len(decisions) != len(pending):
            result["issues"].append("user_source_decisions_coverage_mismatch")
        known = {
            item.get("source_id")
            if isinstance(item, Mapping)
            else getattr(item, "source_id", ""): item
            for item in pending
        }
        seen: set[str] = set()
        normalized: list[dict[str, Any]] = []
        required_keys = {"source_id", "segments"}
        segment_keys = {"exact_span", "decision", "continuation_requirement_ids"}
        requirements = tuple(self._origin_requirements_provider() or ())
        requirement_ids = {str(item.id) for item in requirements}
        for index, item in enumerate(decisions):
            if not isinstance(item, Mapping):
                result["issues"].append(f"user_source_decisions[{index}]_not_mapping")
                continue
            if set(item) != required_keys:
                result["issues"].append(
                    f"user_source_decisions[{index}]_fields_invalid"
                )
            source_id = item.get("source_id")
            segments = item.get("segments")
            if not isinstance(source_id, str) or source_id not in known:
                result["issues"].append(
                    f"user_source_decisions[{index}]_source_invalid"
                )
                continue
            if source_id in seen:
                result["issues"].append(
                    f"user_source_decisions[{index}]_duplicate_source"
                )
            seen.add(source_id)
            if not isinstance(segments, list) or not segments:
                result["issues"].append(
                    f"user_source_decisions[{index}]_segments_invalid"
                )
                segments = []
            source_item = known[source_id]
            source_text = str(
                source_item.get("text", "")
                if isinstance(source_item, Mapping)
                else getattr(source_item, "text", "")
            )
            normalized_segments: list[dict[str, Any]] = []
            spans: list[str] = []
            for segment_index, segment in enumerate(segments):
                if not isinstance(segment, Mapping) or set(segment) != segment_keys:
                    result["issues"].append(
                        f"user_source_decisions[{index}]_segment_{segment_index}_fields_invalid"
                    )
                    continue
                span = segment.get("exact_span")
                segment_decision = segment.get("decision")
                continuation_ids = segment.get("continuation_requirement_ids")
                if not isinstance(span, str) or not span:
                    result["issues"].append(
                        f"user_source_decisions[{index}]_segment_{segment_index}_span_invalid"
                    )
                else:
                    spans.append(span)
                if segment_decision not in {"new_requirement", "continuation_evidence"}:
                    result["issues"].append(
                        f"user_source_decisions[{index}]_segment_{segment_index}_decision_invalid"
                    )
                if not isinstance(continuation_ids, list) or any(
                    (
                        not isinstance(req_id, str) or not req_id
                        for req_id in continuation_ids
                    )
                ):
                    result["issues"].append(
                        f"user_source_decisions[{index}]_segment_{segment_index}_ids_invalid"
                    )
                    continuation_ids = []
                if len(set(continuation_ids)) != len(continuation_ids):
                    result["issues"].append(
                        f"user_source_decisions[{index}]_segment_{segment_index}_duplicate_id"
                    )
                if any((req_id not in requirement_ids for req_id in continuation_ids)):
                    result["issues"].append(
                        f"user_source_decisions[{index}]_segment_{segment_index}_unknown_id"
                    )
                if segment_decision == "new_requirement" and continuation_ids:
                    result["issues"].append(
                        f"user_source_decisions[{index}]_segment_{segment_index}_new_has_ids"
                    )
                if segment_decision == "continuation_evidence" and (
                    not continuation_ids
                ):
                    result["issues"].append(
                        f"user_source_decisions[{index}]_segment_{segment_index}_continuation_unbound"
                    )
                normalized_segments.append(
                    {
                        "exact_span": span,
                        "decision": segment_decision,
                        "continuation_requirement_ids": list(continuation_ids),
                    }
                )
            span_issue = _validate_exact_span_coverage(source_text, spans)
            if span_issue:
                result["issues"].append(f"user_source_decisions[{index}]_{span_issue}")
            normalized.append({"source_id": source_id, "segments": normalized_segments})
        if set(seen) != set(pending_ids):
            result["issues"].append(
                "user_source_decisions_pending_source_coverage_incomplete"
            )
        if result["issues"] or not callable(self._dynamic_requirements_handler):
            if not callable(self._dynamic_requirements_handler):
                result["issues"].append("handler_unavailable")
            self.last_dynamic_requirements = result
            return result
        try:
            appended = tuple(self._dynamic_requirements_handler(tuple(normalized)))
        except Exception as error:
            result["issues"].append(
                f"decision_rejected:{type(error).__name__}:{str(error)[:160]}"
            )
            self.last_dynamic_requirements = result
            return result
        result.update(
            {
                "status": "resolved",
                "decision_status": "resolved",
                "pending_source_ids": [],
                "appended": [getattr(item, "id", str(item)) for item in appended],
            }
        )
        self.last_dynamic_requirements = result
        return result

    @property
    def calls(self) -> int:
        return int(getattr(self.delegate, "calls", 0))

    @property
    def client(self) -> Any:
        return getattr(self.delegate, "client", None)

    def _delegate_propose_with_origin_context(
        self, context: ConceptContext
    ) -> ConceptProposal:
        client = getattr(self.delegate, "client", None)
        original_complete = getattr(client, "complete_json", None)
        if callable(original_complete):
            origin_context = self._origin_context(context)
            system_contract = "ACG structured action-origin contract: return records, action_origins, and user_source_decisions with the required schema. For every pending user source, emit exactly one decision object with exactly source_id and segments; each segment must have exactly exact_span, decision, and continuation_requirement_ids. Copy exact_span text literally from that source, cover the entire source in source order with non-overlapping segments, and use only new_requirement or continuation_evidence. A new segment must have an empty continuation id list; a continuation segment must name existing requirement ids. Do not use source_id, text, unit_id, segment_id, start/end offsets, or alternate field names inside segments."

            def complete_with_origin_context(*args: Any, **kwargs: Any) -> Any:
                request = dict(kwargs)
                request["system_prompt"] = (
                    str(request.get("system_prompt") or "").rstrip()
                    + "\n\n"
                    + system_contract
                )
                request["user_prompt"] = (
                    str(request.get("user_prompt") or "").rstrip()
                    + "\n\n"
                    + origin_context
                )
                return original_complete(*args, **request)

            try:
                setattr(client, "complete_json", complete_with_origin_context)
                proposal = self.delegate.propose(context)
            finally:
                setattr(client, "complete_json", original_complete)
        else:
            proposal = self.delegate.propose(context)
        return _recover_action_origins_payload(proposal, context)

    def propose(self, context: ConceptContext) -> ConceptProposal:
        proposal = self._delegate_propose_with_origin_context(context)
        if not proposal.parse_errors:
            self.last_proposal = proposal
            return proposal
        if _is_semantic_rejection(proposal.parse_errors):
            result = _annotate_unrepaired(
                proposal,
                reason="semantic_parse_rejected",
                outcome=OUTCOME_SEMANTIC_REJECTED,
                repair_attempted=False,
            )
            self.last_proposal = result
            return result
        if not isinstance(proposal.raw_output, str):
            result = _annotate_unrepaired(
                proposal,
                reason="non_string_structural_output",
                outcome=OUTCOME_STRUCTURAL_UNREPAIRABLE,
                repair_attempted=False,
            )
            self.last_proposal = result
            return result
        self.repair_attempts += 1
        repair = repair_missing_new_anchors_brackets(
            proposal.raw_output, context=context
        )
        if repair is None:
            if _looks_like_truncated_response(proposal):
                retry_proposal, retry_metadata = self._retry_truncated(context)
                retry_output = (
                    retry_proposal.raw_output if retry_proposal is not None else None
                )
                if retry_metadata.get("attempted"):
                    if retry_proposal is not None and (not retry_proposal.parse_errors):
                        self.truncation_retries_applied += 1
                        retry_metadata["status"] = "applied"
                        retry_metadata["retry_parse_errors"] = []
                        retry_raw_output: dict[str, Any] = (
                            dict(retry_proposal.raw_output)
                            if isinstance(retry_proposal.raw_output, Mapping)
                            else {}
                        )
                        retry_raw_output.update(
                            {
                                "_projection_repair": {
                                    "status": "retry_applied",
                                    "outcome": OUTCOME_TRUNCATION_RETRY,
                                    "repair_attempted": True,
                                    "repair_applied": False,
                                    "reason": "truncation_retry_succeeded",
                                    "original_parse_errors": list(
                                        proposal.parse_errors
                                    ),
                                    "repaired_parse_errors": [],
                                    "truncation_retry": retry_metadata,
                                },
                                "original_output": proposal.raw_output,
                                "repaired_output": None,
                                "retry_output": retry_output,
                            }
                        )
                        result = ConceptProposal(
                            assignments=retry_proposal.assignments,
                            parse_errors=(),
                            raw_output=retry_raw_output,
                        )
                        self.last_proposal = result
                        return result
                    retry_metadata["status"] = "failed"
                    result = _annotate_unrepaired(
                        proposal,
                        reason="truncation_retry_failed",
                        outcome=OUTCOME_TRUNCATION_RETRY,
                        retry_metadata=retry_metadata,
                        retry_output=retry_output,
                    )
                    self.last_proposal = result
                    return result
                retry_metadata["status"] = "not_attempted"
                result = _annotate_unrepaired(
                    proposal,
                    reason="truncation_retry_not_attempted",
                    outcome=OUTCOME_TRUNCATION_RETRY,
                    retry_metadata=retry_metadata,
                    retry_output=retry_output,
                )
                self.last_proposal = result
                return result
            result = _annotate_unrepaired(
                proposal, reason="no_unique_safe_bracket_repair"
            )
            self.last_proposal = result
            return result
        repaired_proposal = _recover_action_origins_payload(
            parse_concept_output(repair.payload, context=context), context
        )
        if repaired_proposal.parse_errors:
            result = _annotate_unrepaired(
                proposal,
                reason="strict_reparse_failed",
                outcome=_unrepaired_outcome(repaired_proposal.parse_errors),
                repaired_output=repair.payload,
                repaired_parse_errors=repaired_proposal.parse_errors,
            )
            self.last_proposal = result
            return result
        self.repairs_applied += 1
        repair_metadata = {
            "status": "applied",
            "outcome": OUTCOME_SYNTAX_SAFE_REPAIRED,
            "repair_attempted": True,
            "repair_applied": True,
            "kind": "missing_new_anchors_outer_bracket",
            "inserted_brackets": repair.inserted_brackets,
            "positions": list(repair.positions),
            "original_parse_errors": list(proposal.parse_errors),
            "repaired_parse_errors": list(repaired_proposal.parse_errors),
        }
        repaired_raw_output: dict[str, Any] = (
            dict(repaired_proposal.raw_output)
            if isinstance(repaired_proposal.raw_output, Mapping)
            else {}
        )
        repaired_raw_output.update(
            {
                "_projection_repair": repair_metadata,
                "original_output": proposal.raw_output,
                "repaired_output": repair.payload,
                "retry_output": None,
            }
        )
        result = ConceptProposal(
            assignments=repaired_proposal.assignments,
            parse_errors=(),
            raw_output=repaired_raw_output,
        )
        self.last_proposal = result
        return result

    def _retry_truncated(
        self, context: Any
    ) -> tuple[ConceptProposal | None, dict[str, Any]]:
        client = self.client
        original_max_tokens = getattr(client, "max_tokens", None)
        metadata: dict[str, Any] = {
            "attempted": False,
            "applied": False,
            "max_tokens": self.truncation_retry_max_tokens,
            "original_max_tokens": original_max_tokens,
            "reason": None,
        }
        if not isinstance(original_max_tokens, int) or isinstance(
            original_max_tokens, bool
        ):
            metadata["reason"] = "client_max_tokens_unavailable"
            return (None, metadata)
        if self.truncation_retry_max_tokens <= original_max_tokens:
            metadata["reason"] = "retry_budget_not_larger"
            return (None, metadata)
        self.truncation_retry_attempts += 1
        metadata["attempted"] = True
        before = _numeric_client_snapshot(client)
        delegate_calls_before = _numeric_value(self.delegate, "calls")
        retry_proposal: ConceptProposal | None = None
        try:
            client.max_tokens = self.truncation_retry_max_tokens
            retry_proposal = self._delegate_propose_with_origin_context(context)
            metadata["retry_parse_errors"] = list(retry_proposal.parse_errors)
        except Exception as error:
            metadata["error"] = f"{type(error).__name__}: {error}"
        finally:
            client.max_tokens = original_max_tokens
        after = _numeric_client_snapshot(client)
        metadata["delegate_calls_before"] = delegate_calls_before
        metadata["delegate_calls_after"] = _numeric_value(self.delegate, "calls")
        metadata["client_usage_before"] = before
        metadata["client_usage_after"] = after
        metadata["client_usage_delta"] = {
            key: after[key] - before[key] for key in after.keys() & before.keys()
        }
        metadata.setdefault(
            "retry_parse_errors",
            list(retry_proposal.parse_errors) if retry_proposal is not None else [],
        )
        return (retry_proposal, metadata)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)


def _annotate_unrepaired(
    proposal: ConceptProposal,
    *,
    reason: str,
    outcome: str | None = None,
    repair_attempted: bool = True,
    repaired_output: Any = None,
    repaired_parse_errors: tuple[str, ...] = (),
    retry_metadata: Mapping[str, Any] | None = None,
    retry_output: Any = None,
) -> ConceptProposal:
    decision: dict[str, Any] = {
        "status": "not_applied",
        "outcome": outcome or _unrepaired_outcome(proposal.parse_errors),
        "repair_attempted": repair_attempted,
        "repair_applied": False,
        "reason": reason,
        "original_parse_errors": list(proposal.parse_errors),
        "repaired_parse_errors": list(repaired_parse_errors),
    }
    if retry_metadata is not None:
        decision["truncation_retry"] = dict(retry_metadata)
    raw_output: dict[str, Any] = {
        "_projection_repair": decision,
        "original_output": proposal.raw_output,
        "repaired_output": repaired_output,
        "retry_output": retry_output,
    }
    return ConceptProposal(
        assignments=proposal.assignments,
        parse_errors=proposal.parse_errors,
        raw_output=raw_output,
    )


def _looks_like_truncated_response(proposal: ConceptProposal) -> bool:
    raw_output = proposal.raw_output
    if not proposal.parse_errors or not isinstance(raw_output, str):
        return False
    return _has_unterminated_json_string(raw_output)


def _has_unterminated_json_string(payload: str) -> bool:
    in_string = False
    escaped = False
    for char in payload:
        if not in_string:
            if char == '"':
                in_string = True
            continue
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            in_string = False
    return in_string


def _numeric_value(obj: Any, name: str) -> int | None:
    value = getattr(obj, name, None)
    if isinstance(value, int) and (not isinstance(value, bool)):
        return value
    return None


def _numeric_client_snapshot(client: Any) -> dict[str, int]:
    names = (
        "calls",
        "failures",
        "usage_records",
        "prompt_tokens",
        "generation_tokens",
        "cached_tokens",
    )
    return {
        name: value
        for name in names
        if (value := _numeric_value(client, name)) is not None
    }


_AMBIGUOUS = object()


def _skip_space(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _consume_string(text: str, start: int) -> int | None:
    index = start + 1
    while index < len(text):
        char = text[index]
        if char == "\\":
            if index + 1 >= len(text):
                return None
            index += 2
            continue
        if char == '"':
            return index + 1
        index += 1
    return None


def _string_key_spans(text: str, *, key: str) -> list[tuple[int, int]] | None:
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        if text[index] != '"':
            index += 1
            continue
        end = _consume_string(text, index)
        if end is None:
            return None
        try:
            value = json.loads(text[index:end])
        except json.JSONDecodeError:
            return None
        if value == key:
            spans.append((index, end))
        index = end
    return spans


def _scan_anchor_array(text: str, start: int) -> int | object | None:
    depth = 1
    saw_nested_array = False
    index = start + 1
    while index < len(text):
        char = text[index]
        if char == '"':
            end = _consume_string(text, index)
            if end is None:
                return _AMBIGUOUS
            index = end
            continue
        if char == "[":
            depth += 1
            saw_nested_array = True
        elif char == "]":
            depth -= 1
            if depth < 0:
                return _AMBIGUOUS
            if depth == 0:
                return None
        elif char == "{":
            if depth >= 1:
                return _AMBIGUOUS
        elif char == "}" and depth >= 1:
            previous = _previous_non_space(text, index)
            following = _skip_space(text, index + 1)
            if (
                depth != 1
                or not saw_nested_array
                or previous >= len(text)
                or (text[previous] != "]")
                or (following >= len(text))
                or (text[following] not in ",]")
            ):
                return _AMBIGUOUS
            return index
        index += 1
    return _AMBIGUOUS


def _previous_non_space(text: str, index: int) -> int:
    index -= 1
    while index >= 0 and text[index].isspace():
        index -= 1
    return index
