from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .evidence import EvidenceRecord, EvidenceUnit, extract_complete_evidence_records


def extract_record_level_evidence_records(
    text: str, *, id_prefix: str = ""
) -> tuple[EvidenceRecord, ...]:
    try:
        structured = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return extract_complete_evidence_records(text, id_prefix=id_prefix)
    residual, recovered, changed = _recover_serialized_records(structured, path="")
    if not changed:
        return extract_complete_evidence_records(text, id_prefix=id_prefix)
    rooted_records = list(recovered)
    if residual is not _DROP:
        rooted_records.extend(_rooted_record_values(residual, path=""))
    return tuple(
        (
            _make_record(index, path, value, id_prefix=id_prefix)
            for index, (path, value) in enumerate(rooted_records, start=1)
        )
    )


_DROP = object()


def _recover_serialized_records(
    value: Any, *, path: str
) -> tuple[Any, tuple[tuple[str, Mapping[str, Any]], ...], bool]:
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate or candidate[0] not in "[{":
            return (value, (), False)
        try:
            decoded = json.loads(candidate)
        except json.JSONDecodeError:
            return (value, (), False)
        if not isinstance(decoded, (Mapping, list)):
            return (value, (), False)
        return (_DROP, _records_from_decoded_container(decoded, path=path), True)
    if isinstance(value, Mapping):
        changed = False
        residual_mapping: dict[str, Any] = {}
        records: list[tuple[str, Mapping[str, Any]]] = []
        for key, child in value.items():
            child_path = f"{path}/{_json_pointer_token(str(key))}"
            residual_child, child_records, child_changed = _recover_serialized_records(
                child, path=child_path
            )
            if residual_child is not _DROP:
                residual_mapping[str(key)] = residual_child
            records.extend(child_records)
            changed = changed or child_changed
        return (residual_mapping, tuple(records), changed)
    if isinstance(value, list):
        changed = False
        residual_items: dict[str, Any] = {}
        records: list[tuple[str, Mapping[str, Any]]] = []
        for index, child in enumerate(value):
            residual_child, child_records, child_changed = _recover_serialized_records(
                child, path=f"{path}/{index}"
            )
            if residual_child is not _DROP:
                residual_items[str(index)] = residual_child
            records.extend(child_records)
            changed = changed or child_changed
        return (residual_items, tuple(records), changed)
    return (value, (), False)


def _records_from_decoded_container(
    value: Any, *, path: str
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    if (
        isinstance(value, list)
        and value
        and all((isinstance(item, Mapping) for item in value))
    ):
        records: list[tuple[str, Mapping[str, Any]]] = []
        for index, item in enumerate(value):
            item_path = f"{path}/{index}"
            residual, nested_records, _ = _recover_serialized_records(
                item, path=item_path
            )
            records.extend(nested_records)
            if residual is not _DROP and residual:
                records.append((item_path, residual))
        return tuple(records)
    residual, nested_records, _ = _recover_serialized_records(value, path=path)
    records = list(nested_records)
    if residual is not _DROP:
        records.extend(_rooted_record_values(residual, path=path))
    return tuple(records)


def _rooted_record_values(
    value: Any, *, path: str
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    if isinstance(value, list):
        if value and all((isinstance(item, Mapping) for item in value)):
            return tuple(
                ((f"{path}/{index}", item) for index, item in enumerate(value))
            )
        return ((path, {"value": value}),)
    if not isinstance(value, Mapping):
        return ((path, {"value": value}),)
    records, residual = _split_first_mapping_list(value, path=path)
    if not records:
        return ((path, value),)
    if residual:
        return (*records, (path, residual))
    return records


def _split_first_mapping_list(
    value: Mapping[str, Any], *, path: str
) -> tuple[tuple[tuple[str, Mapping[str, Any]], ...], dict[str, Any]]:
    for key, child in value.items():
        child_path = f"{path}/{_json_pointer_token(str(key))}"
        if (
            isinstance(child, list)
            and child
            and all((isinstance(item, Mapping) for item in child))
        ):
            records = tuple(
                ((f"{child_path}/{index}", item) for index, item in enumerate(child))
            )
            residual = {
                other_key: other
                for other_key, other in value.items()
                if other_key != key
            }
            return (records, residual)
        if isinstance(child, Mapping):
            records, child_residual = _split_first_mapping_list(child, path=child_path)
            if records:
                residual = dict(value)
                if child_residual:
                    residual[key] = child_residual
                else:
                    residual.pop(key)
                return (records, residual)
    return ((), dict(value))


def _make_record(
    index: int, path: str, value: Mapping[str, Any], *, id_prefix: str
) -> EvidenceRecord:
    record_id = f"{(id_prefix + '.' if id_prefix else '')}record_{index}"
    units = tuple(
        (
            EvidenceUnit(f"{record_id}.unit_{unit_index}", unit_path, item)
            for unit_index, (unit_path, item) in enumerate(
                _flatten(value, path=path), start=1
            )
        )
    )
    return EvidenceRecord(record_id, units)


def _flatten(value: Any, *, path: str) -> tuple[tuple[str, Any], ...]:
    if isinstance(value, Mapping):
        return tuple(
            (
                item
                for key, child in value.items()
                for item in _flatten(
                    child, path=f"{path}/{_json_pointer_token(str(key))}"
                )
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
                for item in _flatten(child, path=f"{path}/{index}")
            )
        )
    return ((path or "/", value),)


def _json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


__all__ = ["extract_record_level_evidence_records"]
