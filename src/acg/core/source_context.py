from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence

from .store import source_payload_text

if TYPE_CHECKING:
    from . import AdaptiveConsistencyGraph


@dataclass(frozen=True)
class TaskSource:
    source_id: str
    start_offset: int
    end_offset: int
    text: str


@dataclass(frozen=True)
class RequirementAnchor:
    id: str
    source_id: str
    start_offset: int
    end_offset: int
    exact_text: str
    label: str = ""


@dataclass(frozen=True)
class UserSourceSegment:
    exact_span: str
    decision: str
    continuation_requirement_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class UserEvidence:
    source_id: str
    sequence: int
    text: str
    status: str = "pending"
    decision: str | None = None
    exact_spans: tuple[str, ...] = ()
    continuation_requirement_ids: tuple[str, ...] = ()
    segments: tuple[UserSourceSegment, ...] = ()
    preceding_assistant_source_id: str | None = None
    preceding_assistant_text: str | None = None


USER_SOURCE_PENDING = "pending"
USER_SOURCE_RESOLVED = "resolved"
USER_SOURCE_DECISIONS = frozenset({"new_requirement", "continuation_evidence"})


class TaskContractClient(Protocol):
    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: Mapping[str, Any],
        temperature: float | None = None,
    ) -> str:
        raise NotImplementedError


@dataclass
class SourceState:
    task_source: TaskSource | None = None
    task_contracts: tuple[RequirementAnchor, ...] = ()
    task_contract_status: dict[str, Any] = field(default_factory=dict)
    user_evidence: list[UserEvidence] = field(default_factory=list)
    later_user_sources: set[str] = field(default_factory=set)


def _source_state(plugin: Any) -> SourceState:
    state = getattr(plugin, "projection_source_state", None)
    if not isinstance(state, SourceState):
        state = SourceState()
        setattr(plugin, "projection_source_state", state)
    return state


_task_envelopes: dict[str, str] = {}
TASK_CONTRACT_SYSTEM_PROMPT = "You split a user task into source-faithful requirement anchors.\n\nYou are not planning, judging completion, ranking options, or deciding actions.\nReturn a JSON object containing only exact spans copied from the original task. A span should represent one\nindependent requirement that a future executor must consider. Do not add implied\nconstraints, domain fields, statuses, dependencies, or explanations."
TASK_CONTRACT_USER_PROMPT = "Original task:\n<<<TASK\n{task}\nTASK>>>\n\nReturn the requirement anchors as a JSON object with a `requirements` array. Each\nrequirement anchor must be one of the exact substrings of the original task. If the\ntask already contains a single atomic requirement, return one span containing the\nwhole task. Labels must be short and descriptive but must not add information that\nis absent from the exact span."
TASK_CONTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "requirements": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "exact_span": {"type": "string", "minLength": 1},
                    "label": {"type": "string"},
                },
                "required": ["exact_span", "label"],
            },
        }
    },
    "required": ["requirements"],
}


def register_task_envelope(envelope: str, task_text: str) -> None:
    envelope = str(envelope)
    task_text = str(task_text).strip()
    if not task_text:
        raise ValueError("Task source text must not be empty")
    if task_text not in envelope:
        raise ValueError("Task source text must be an exact span of the task envelope")
    existing = _task_envelopes.get(envelope)
    if existing is not None and existing != task_text:
        raise ValueError("A task envelope cannot map to two different task sources")
    _task_envelopes[envelope] = task_text


def register_task_source(
    plugin: AdaptiveConsistencyGraph, text: str, *, source_id: str | None = None
) -> TaskSource:
    source = plugin.task if source_id is None else plugin.sources.get(source_id)
    exact_text = str(text).strip()
    if not exact_text:
        raise ValueError("Task source text must not be empty")
    if exact_text not in source_payload_text(source):
        raise ValueError("Task source text must be grounded in its source record")
    source_text = source_payload_text(source)
    start_offset = source_text.index(exact_text)
    registered = TaskSource(
        source_id=source.id,
        start_offset=start_offset,
        end_offset=start_offset + len(exact_text),
        text=exact_text,
    )
    _source_state(plugin).task_source = registered
    return registered


def task_source(plugin: AdaptiveConsistencyGraph) -> TaskSource:
    registered = _source_state(plugin).task_source
    if registered:
        return registered
    envelope = source_payload_text(plugin.task)
    return register_task_source(
        plugin, _task_envelopes.get(envelope, envelope), source_id=plugin.task.id
    )


def register_user_evidence(
    plugin: AdaptiveConsistencyGraph,
    *,
    source_id: str,
    text: str,
    preceding_assistant_source_id: str | None = None,
    preceding_assistant_text: str | None = None,
) -> UserEvidence:
    source = plugin.sources.get(source_id)
    exact_text = str(text).strip()
    if not exact_text:
        raise ValueError("User evidence text must not be empty")
    if exact_text not in source_payload_text(source):
        raise ValueError("User evidence must be an exact span of its source record")
    item = UserEvidence(
        source_id=source.id,
        sequence=source.sequence,
        text=exact_text,
        preceding_assistant_source_id=str(preceding_assistant_source_id)
        if preceding_assistant_source_id is not None
        else None,
        preceding_assistant_text=str(preceding_assistant_text)
        if preceding_assistant_text is not None
        else None,
    )
    items = _source_state(plugin).user_evidence
    for existing in items:
        if existing.source_id == item.source_id:
            return existing
    if all((existing.source_id != item.source_id for existing in items)):
        items.append(item)
        items.sort(key=lambda existing: existing.sequence)
    _source_state(plugin).later_user_sources.add(item.source_id)
    return item


def later_user_evidence(plugin: AdaptiveConsistencyGraph) -> tuple[UserEvidence, ...]:
    return tuple(reversed(_source_state(plugin).user_evidence))


def pending_later_user_evidence(
    plugin: AdaptiveConsistencyGraph,
) -> tuple[UserEvidence, ...]:
    return tuple(
        (
            item
            for item in later_user_evidence(plugin)
            if item.status == USER_SOURCE_PENDING
        )
    )


def resolved_later_user_evidence(
    plugin: AdaptiveConsistencyGraph,
) -> tuple[UserEvidence, ...]:
    return tuple(
        (
            item
            for item in later_user_evidence(plugin)
            if item.status == USER_SOURCE_RESOLVED
        )
    )


def user_source_status(
    plugin: AdaptiveConsistencyGraph, source_id: str
) -> Mapping[str, Any]:
    for item in _source_state(plugin).user_evidence:
        if item.source_id == str(source_id):
            return {
                "source_id": item.source_id,
                "sequence": item.sequence,
                "text": item.text,
                "status": item.status,
                "decision": item.decision,
                "exact_spans": list(item.exact_spans),
                "continuation_requirement_ids": list(item.continuation_requirement_ids),
                "segments": [
                    {
                        "exact_span": segment.exact_span,
                        "decision": segment.decision,
                        "continuation_requirement_ids": list(
                            segment.continuation_requirement_ids
                        ),
                    }
                    for segment in item.segments
                ],
                "preceding_assistant_source_id": item.preceding_assistant_source_id,
                "preceding_assistant_text": item.preceding_assistant_text,
            }
    return {"source_id": str(source_id), "status": "unknown"}


def resolve_later_user_source(
    plugin: AdaptiveConsistencyGraph,
    *,
    source_id: str,
    decision: str,
    exact_spans: Sequence[str] = (),
    continuation_requirement_ids: Sequence[str] = (),
    segments: Sequence[UserSourceSegment | Mapping[str, Any]] = (),
) -> UserEvidence:
    normalized_source_id = str(source_id)
    raw_segments = tuple(segments)
    normalized_segments: list[UserSourceSegment] = []
    if raw_segments:
        for item in raw_segments:
            if isinstance(item, UserSourceSegment):
                segment = item
            elif isinstance(item, Mapping):
                segment = UserSourceSegment(
                    exact_span=item.get("exact_span"),
                    decision=item.get("decision"),
                    continuation_requirement_ids=tuple(
                        item.get("continuation_requirement_ids") or ()
                    ),
                )
            else:
                raise ValueError("Later-user segment must be a mapping")
            if not isinstance(segment.exact_span, str) or not segment.exact_span:
                raise ValueError("Later-user segment exact_span must be non-empty")
            if segment.decision not in USER_SOURCE_DECISIONS:
                raise ValueError("Later-user segment decision is unsupported")
            if (
                segment.decision == "new_requirement"
                and segment.continuation_requirement_ids
            ):
                raise ValueError(
                    "new_requirement segment cannot include requirement IDs"
                )
            if segment.decision == "continuation_evidence" and (
                not segment.continuation_requirement_ids
            ):
                raise ValueError(
                    "continuation segment requires an existing requirement ID"
                )
            normalized_segments.append(segment)
        spans = tuple((item.exact_span for item in normalized_segments))
        continuation_ids = tuple(
            dict.fromkeys(
                (
                    req_id
                    for item in normalized_segments
                    for req_id in item.continuation_requirement_ids
                )
            )
        )
        decision = (
            normalized_segments[0].decision
            if len({item.decision for item in normalized_segments}) == 1
            else "mixed"
        )
    else:
        spans = tuple(exact_spans)
        continuation_ids = tuple(continuation_requirement_ids)
        normalized_segments = [
            UserSourceSegment(
                exact_span=span,
                decision=decision,
                continuation_requirement_ids=continuation_ids,
            )
            for span in spans
        ]
    if decision not in USER_SOURCE_DECISIONS and (not raw_segments):
        raise ValueError(f"Unsupported later-user decision: {decision!r}")
    if any((not isinstance(item, str) or not item for item in spans)):
        raise ValueError("Later-user exact_spans must contain non-empty strings")
    if any((not isinstance(item, str) or not item for item in continuation_ids)):
        raise ValueError("Later-user continuation_requirement_ids must contain strings")
    if len(set(continuation_ids)) != len(continuation_ids):
        raise ValueError("Later-user continuation_requirement_ids must be unique")
    if not raw_segments and decision == "new_requirement" and continuation_ids:
        raise ValueError("new_requirement cannot include continuation requirement IDs")
    if (
        not raw_segments
        and decision == "continuation_evidence"
        and (not continuation_ids)
    ):
        raise ValueError("continuation_evidence requires an existing requirement ID")
    existing_requirements = tuple(
        _source_state(plugin).task_contracts
        or source_grounded_requirement_anchors(plugin)
    )
    known_ids = {item.id for item in existing_requirements}
    if any((item not in known_ids for item in continuation_ids)):
        raise ValueError("Continuation requirement ID is not in the current catalog")
    items = _source_state(plugin).user_evidence
    for index, item in enumerate(items):
        if item.source_id != normalized_source_id:
            continue
        if item.status != USER_SOURCE_PENDING:
            raise ValueError("Later-user source is already resolved")
        resolved = UserEvidence(
            source_id=item.source_id,
            sequence=item.sequence,
            text=item.text,
            status=USER_SOURCE_RESOLVED,
            decision=decision,
            exact_spans=spans,
            continuation_requirement_ids=continuation_ids,
            segments=tuple(normalized_segments),
            preceding_assistant_source_id=item.preceding_assistant_source_id,
            preceding_assistant_text=item.preceding_assistant_text,
        )
        items[index] = resolved
        return resolved
    raise ValueError("Later-user source is not registered")


def append_dynamic_requirements(
    plugin: AdaptiveConsistencyGraph,
    *,
    source_id: str,
    exact_spans: Sequence[Any] | Any = (),
) -> tuple[RequirementAnchor, ...]:
    if exact_spans in (None, ()):
        return ()
    if isinstance(exact_spans, (str, bytes)) or not isinstance(exact_spans, Sequence):
        exact_spans = (exact_spans,)
    spans = tuple(exact_spans)
    if not spans:
        return ()
    normalized_source_id = str(source_id)
    if normalized_source_id not in _source_state(plugin).later_user_sources:
        raise ValueError("Dynamic requirements require a registered later-user source")
    source = plugin.sources.get(normalized_source_id)
    if source is None:
        raise ValueError("Dynamic requirement source is unknown")
    source_text = source_payload_text(source)
    existing = tuple(
        _source_state(plugin).task_contracts
        or source_grounded_requirement_anchors(plugin)
    )
    occupied: list[tuple[int, int]] = [
        (item.start_offset, item.end_offset)
        for item in existing
        if item.source_id == normalized_source_id
    ]
    seen: set[tuple[int, int]] = set()
    pending: list[RequirementAnchor] = []
    for item in spans:
        if isinstance(item, str):
            exact_span = item
            label = ""
        elif isinstance(item, Mapping):
            exact_span = item.get("exact_span")
            label = " ".join(str(item.get("label") or "").split())
        else:
            raise ValueError("Dynamic requirement must contain an exact_span string")
        if not isinstance(exact_span, str) or not exact_span:
            raise ValueError(
                "Dynamic requirement exact_span must be a non-empty string"
            )
        start = source_text.find(exact_span)
        end = start + len(exact_span) if start >= 0 else -1
        if start < 0:
            raise ValueError(
                "Dynamic requirement exact_span is not grounded in its source"
            )
        interval = (start, end)
        if interval in seen or any(
            (not (end <= left or start >= right) for left, right in occupied)
        ):
            raise ValueError(
                "Dynamic requirement exact_span is duplicate or overlapping"
            )
        seen.add(interval)
        occupied.append(interval)
        pending.append(
            RequirementAnchor(
                id=f"R{len(existing) + len(pending) + 1}",
                source_id=normalized_source_id,
                start_offset=start,
                end_offset=end,
                exact_text=exact_span,
                label=label,
            )
        )
    anchors = (*existing, *pending)
    state = _source_state(plugin)
    state.task_contracts = anchors
    status = dict(state.task_contract_status)
    status.update(
        {
            "status": "registered",
            "source": status.get("source", "append_only_dynamic_exact_span"),
            "n_requirements": len(anchors),
            "dynamic_requirements_appended": len(pending),
        }
    )
    state.task_contract_status = status
    return tuple(pending)


def dynamic_requirement_anchors(
    plugin: AdaptiveConsistencyGraph,
) -> tuple[RequirementAnchor, ...]:
    state = _source_state(plugin)
    later_sources = state.later_user_sources
    return tuple(
        (item for item in state.task_contracts or () if item.source_id in later_sources)
    )


def requirement_anchors(
    plugin: AdaptiveConsistencyGraph,
) -> tuple[RequirementAnchor, ...]:
    registered = _source_state(plugin).task_contracts
    if registered is not None:
        return registered
    source = task_source(plugin)
    spans = _structural_spans(source.text)
    return tuple(
        (
            RequirementAnchor(
                id=f"R{index}",
                source_id=source.source_id,
                start_offset=source.start_offset + start,
                end_offset=source.start_offset + end,
                exact_text=source.text[start:end],
            )
            for index, (start, end) in enumerate(spans, start=1)
        )
    )


def register_task_contract(
    plugin: AdaptiveConsistencyGraph,
    requirements: Sequence[Any],
    *,
    source: TaskSource | None = None,
) -> tuple[RequirementAnchor, ...]:
    task = source or task_source(plugin)
    anchors = _validated_requirement_anchors(task, requirements)
    if not anchors:
        raise ValueError(
            "Task contract did not contain any valid exact-span requirements"
        )
    state = _source_state(plugin)
    state.task_contracts = anchors
    state.task_contract_status = {
        "status": "registered",
        "source": "provided",
        "n_requirements": len(anchors),
    }
    return anchors


def register_task_contract_from_client(
    plugin: AdaptiveConsistencyGraph,
    client: TaskContractClient | None,
    *,
    max_requirements: int = 8,
) -> tuple[RequirementAnchor, ...]:
    state = _source_state(plugin)
    if state.task_contracts:
        return state.task_contracts
    source = task_source(plugin)
    if client is None:
        state.task_contract_status = {"status": "fallback", "reason": "missing_client"}
        return requirement_anchors(plugin)
    try:
        raw = client.complete_json(
            system_prompt=TASK_CONTRACT_SYSTEM_PROMPT,
            user_prompt=TASK_CONTRACT_USER_PROMPT.format(task=source.text),
            schema=TASK_CONTRACT_SCHEMA,
            temperature=0.0,
        )
        parsed = _loads_json_object(raw)
        items = parsed.get("requirements")
        if not isinstance(items, list):
            raise ValueError("Task contract response missing requirements array")
        anchors = _validated_requirement_anchors(source, items[:max_requirements])
        if not anchors:
            raise ValueError("Task contract response had no valid exact spans")
        state.task_contracts = anchors
        state.task_contract_status = {
            "status": "registered",
            "source": "llm_exact_span",
            "n_requirements": len(anchors),
        }
        return anchors
    except Exception as error:
        state.task_contract_status = {
            "status": "fallback",
            "reason": type(error).__name__,
            "message": str(error)[:240],
        }
        return requirement_anchors(plugin)


def task_contract_status(plugin: AdaptiveConsistencyGraph) -> Mapping[str, Any]:
    state = _source_state(plugin)
    return dict(state.task_contract_status or {"status": "structural_fallback"})


def source_grounded_requirement_anchors(
    plugin: AdaptiveConsistencyGraph,
) -> tuple[RequirementAnchor, ...]:
    anchors = requirement_anchors(plugin)
    if task_contract_status(plugin).get("status") == "registered":
        return anchors
    return (_whole_task_requirement(task_source(plugin)),)


def _whole_task_requirement(source: TaskSource) -> RequirementAnchor:
    return RequirementAnchor(
        id="R1",
        source_id=source.source_id,
        start_offset=source.start_offset,
        end_offset=source.end_offset,
        exact_text=source.text,
    )


def _validated_requirement_anchors(
    source: TaskSource, requirements: Sequence[Any]
) -> tuple[RequirementAnchor, ...]:
    anchors: list[RequirementAnchor] = []
    occupied: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for item in requirements:
        if isinstance(item, str):
            exact_span = item.strip()
            label = ""
        elif isinstance(item, Mapping):
            exact_span = str(
                item.get("exact_span")
                or item.get("exact_text")
                or item.get("span")
                or item.get("text")
                or ""
            ).strip()
            label = " ".join(str(item.get("label") or "").split())
        else:
            continue
        if not exact_span:
            continue
        located = _find_non_overlapping_span(source.text, exact_span, occupied)
        if located is None:
            continue
        start, end = located
        if (start, end) in seen:
            continue
        anchors.append(
            RequirementAnchor(
                id=f"R{len(anchors) + 1}",
                source_id=source.source_id,
                start_offset=source.start_offset + start,
                end_offset=source.start_offset + end,
                exact_text=source.text[start:end],
                label=label,
            )
        )
        occupied.append((start, end))
        occupied.sort()
        seen.add((start, end))
    return tuple(anchors)


def _find_non_overlapping_span(
    text: str, exact_span: str, occupied: Sequence[tuple[int, int]]
) -> tuple[int, int] | None:
    start = 0
    while True:
        index = text.find(exact_span, start)
        if index < 0:
            return _find_normalized_non_overlapping_span(text, exact_span, occupied)
        end = index + len(exact_span)
        if all((end <= left or index >= right for left, right in occupied)):
            return (index, end)
        start = index + 1


def _find_normalized_non_overlapping_span(
    text: str, exact_span: str, occupied: Sequence[tuple[int, int]]
) -> tuple[int, int] | None:
    normalized_span = " ".join(exact_span.split())
    if not normalized_span:
        return None
    target_tokens = normalized_span.split()
    source_tokens = tuple(re.finditer("\\S+", text))
    width = len(target_tokens)
    for left_index in range(len(source_tokens) - width + 1):
        candidate_tokens = [
            source_tokens[index].group(0)
            for index in range(left_index, left_index + width)
        ]
        if candidate_tokens != target_tokens:
            continue
        start = source_tokens[left_index].start()
        end = source_tokens[left_index + width - 1].end()
        if all((end <= left or start >= right for left, right in occupied)):
            return (start, end)
    return None


def _loads_json_object(raw: str) -> dict[str, Any]:
    text = str(raw).strip()
    fence = chr(96) * 3
    if text.startswith(fence):
        text = re.sub("^" + fence + "(?:json)?\\s*", "", text, flags=re.IGNORECASE)
        text = re.sub("\\s*" + fence + "$", "", text)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Task contract response must be a JSON object")
    return data


def _structural_spans(text: str) -> tuple[tuple[int, int], ...]:
    nonempty_lines = tuple(
        (
            match
            for match in re.finditer(
                "(?m)^[ \\t]*(?:[-*]+|\\d+[.)])\\s+(.+?)\\s*$", text
            )
            if match.group(1).strip()
        )
    )
    if len(nonempty_lines) >= 2:
        return tuple(
            (_trim_span(text, match.start(1), match.end(1)) for match in nonempty_lines)
        )
    discourse_starts = tuple(
        (
            match.start()
            for match in re.finditer(
                "(?i)(?:(?<=^)|(?<=[.!?]))\\s*(?:first(?:ly)?|second(?:ly)?|third(?:ly)?|next|then|additionally|furthermore|moreover|(?:i(?:'|’)m\\s+)?also|finally|lastly)\\b",
                text,
            )
        )
    )
    if len(discourse_starts) >= 2:
        boundaries = (0, *discourse_starts[1:], len(text))
        return tuple(
            (
                _trim_span(text, start, end)
                for start, end in zip(boundaries, boundaries[1:])
            )
        )
    semicolons = tuple((match.end() for match in re.finditer(";", text)))
    if len(semicolons) >= 2:
        boundaries = (0, *semicolons, len(text))
        return tuple(
            (
                _trim_span(text, start, end)
                for start, end in zip(boundaries, boundaries[1:])
            )
        )
    return (_trim_span(text, 0, len(text)),)


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end)


__all__ = [
    "RequirementAnchor",
    "TaskSource",
    "UserEvidence",
    "UserSourceSegment",
    "later_user_evidence",
    "pending_later_user_evidence",
    "resolved_later_user_evidence",
    "resolve_later_user_source",
    "register_task_envelope",
    "register_task_source",
    "register_task_contract",
    "register_task_contract_from_client",
    "register_user_evidence",
    "requirement_anchors",
    "source_grounded_requirement_anchors",
    "task_contract_status",
    "task_source",
    "user_source_status",
    "USER_SOURCE_PENDING",
    "USER_SOURCE_RESOLVED",
    "USER_SOURCE_DECISIONS",
    "append_dynamic_requirements",
    "dynamic_requirement_anchors",
]
