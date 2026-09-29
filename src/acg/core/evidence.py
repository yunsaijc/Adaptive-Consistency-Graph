from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

TRUNCATION_PATTERN = re.compile("\\.\\.\\. <truncated \\d+ chars>")


@dataclass(frozen=True)
class EvidenceUnit:
    id: str
    path: str
    value: Any

    @property
    def text(self) -> str:
        if self.path.startswith("/text/") and isinstance(self.value, str):
            return self.value
        return f"{self.path} = {json.dumps(self.value, ensure_ascii=False, sort_keys=True)}"


@dataclass(frozen=True)
class EvidenceRecord:
    id: str
    units: tuple[EvidenceUnit, ...]


def extract_complete_evidence_records(
    text: str, *, id_prefix: str = ""
) -> tuple[EvidenceRecord, ...]:
    structured = _complete_json_value(text)
    if structured is not None:
        values = _record_values(structured)
        return tuple(
            (
                _make_record(index, value, id_prefix=id_prefix)
                for index, value in enumerate(values, start=1)
            )
        )
    recovered = _maximal_complete_objects(text)
    if recovered:
        return tuple(
            (
                _make_record(index, value, id_prefix=id_prefix)
                for index, value in enumerate(recovered, start=1)
            )
        )
    if TRUNCATION_PATTERN.search(text) and _looks_structured(text):
        return ()
    return _complete_text_records(text, id_prefix=id_prefix)


def _complete_json_value(text: str) -> Any | None:
    decoder = json.JSONDecoder()
    for offset, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, consumed = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            continue
        if not text[offset + consumed :].strip():
            return value
    return None


def _looks_structured(text: str) -> bool:
    return bool(
        re.search("[\\[{]", text) and re.search('"(?:[^"\\\\]|\\\\.)*"\\s*:', text)
    )


@dataclass(frozen=True)
class _Container:
    opener: str
    start: int
    object_depth: int
    object_array_element: bool


def _maximal_complete_objects(text: str) -> tuple[Mapping[str, Any], ...]:
    candidates: list[tuple[int, int, int, Mapping[str, Any]]] = []
    stack: list[_Container] = []
    in_string = False
    escaped = False
    for index, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
            continue
        if character in "[{":
            object_depth = sum((container.opener == "{" for container in stack))
            stack.append(
                _Container(
                    opener=character,
                    start=index,
                    object_depth=object_depth + (character == "{"),
                    object_array_element=character == "{"
                    and bool(stack)
                    and (stack[-1].opener == "["),
                )
            )
            continue
        if character not in "]}" or not stack:
            continue
        container = stack[-1]
        if (container.opener, character) not in {("[", "]"), ("{", "}")}:
            continue
        stack.pop()
        if container.opener != "{":
            continue
        if any((ancestor.object_array_element for ancestor in stack)):
            continue
        source = text[container.start : index + 1]
        try:
            value = json.loads(source)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            candidates.append(
                (container.start, index + 1, container.object_depth, value)
            )
    maximal = [
        candidate
        for candidate in candidates
        if not any(
            (
                other[0] <= candidate[0]
                and candidate[1] <= other[1]
                and ((other[0], other[1]) != (candidate[0], candidate[1]))
                for other in candidates
            )
        )
    ]
    if maximal:
        shallowest_depth = min((candidate[2] for candidate in maximal))
        maximal = [
            candidate for candidate in maximal if candidate[2] == shallowest_depth
        ]
    maximal.sort(key=lambda item: item[0])
    return tuple((value for _, _, _, value in maximal))


def _record_values(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, list):
        if value and all((isinstance(item, Mapping) for item in value)):
            return tuple(value)
        return ({"value": value},)
    if not isinstance(value, Mapping):
        return ({"value": value},)
    records, residual = _split_first_mapping_list(value)
    if not records:
        return (value,)
    if residual:
        return (*records, residual)
    return records


def _split_first_mapping_list(
    value: Mapping[str, Any],
) -> tuple[tuple[Mapping[str, Any], ...], dict[str, Any]]:
    for key, child in value.items():
        if (
            isinstance(child, list)
            and child
            and all((isinstance(item, Mapping) for item in child))
        ):
            return (
                tuple(child),
                {
                    other_key: other
                    for other_key, other in value.items()
                    if other_key != key
                },
            )
        if isinstance(child, Mapping):
            records, child_residual = _split_first_mapping_list(child)
            if records:
                residual = dict(value)
                if child_residual:
                    residual[key] = child_residual
                else:
                    residual.pop(key)
                return (records, residual)
    return ((), dict(value))


def _make_record(
    index: int, value: Mapping[str, Any], *, id_prefix: str
) -> EvidenceRecord:
    record_id = _record_id(id_prefix, index)
    units = tuple(
        (
            EvidenceUnit(f"{record_id}.unit_{unit_index}", path, item)
            for unit_index, (path, item) in enumerate(_flatten(value), start=1)
        )
    )
    return EvidenceRecord(record_id, units)


def _flatten(value: Any, path: str = "") -> tuple[tuple[str, Any], ...]:
    if isinstance(value, Mapping):
        return tuple(
            (
                item
                for key, child in value.items()
                for item in _flatten(child, f"{path}/{_json_pointer_token(str(key))}")
            )
        )
    if isinstance(value, list):
        if not value or all(
            (not isinstance(child, (Mapping, list)) for child in value)
        ):
            return ((path or "/", value),)
        return tuple(
            (
                item
                for index, child in enumerate(value)
                for item in _flatten(child, f"{path}/{index}")
            )
        )
    return ((path or "/", value),)


def _complete_text_records(text: str, *, id_prefix: str) -> tuple[EvidenceRecord, ...]:
    marker = TRUNCATION_PATTERN.search(text)
    safe_text = text[: marker.start()] if marker else text
    groups: list[list[str]] = []
    if marker and "\n" in safe_text:
        lines = safe_text.splitlines()
        while lines and (not lines[-1].strip()):
            lines.pop()
        if lines:
            lines.pop()
        complete_lines = [line.strip() for line in lines if line.strip()]
        if complete_lines:
            groups.append(complete_lines)
    elif marker:
        sentences = [
            match.group(0).strip()
            for match in re.finditer("[^.!?]+[.!?](?:\\s+|$)", safe_text)
            if match.group(0).strip()
        ]
        if sentences:
            groups.append(sentences)
    else:
        for paragraph in re.split("\\n\\s*\\n", safe_text):
            lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
            if lines:
                groups.append(lines)
    records = []
    for record_index, group in enumerate(groups, start=1):
        record_id = _record_id(id_prefix, record_index)
        units = tuple(
            (
                EvidenceUnit(
                    f"{record_id}.unit_{unit_index}", f"/text/{unit_index - 1}", line
                )
                for unit_index, line in enumerate(group, start=1)
            )
        )
        records.append(EvidenceRecord(record_id, units))
    return tuple(records)


def _json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _record_id(prefix: str, index: int) -> str:
    return f"{(prefix + '.' if prefix else '')}record_{index}"
