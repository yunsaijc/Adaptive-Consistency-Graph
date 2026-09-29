from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .source_context import RequirementAnchor

LINEAGE_BASES = frozenset(
    {
        "single_requirement_task",
        "exact_action_argument",
        "inherited_object_origin",
        "tool_observation_inheritance",
        "explicit_action_origin_marker",
        "structured_builder_origin",
        "user_source_decision",
    }
)
LINEAGE_SCHEMA_VERSION = "projection.lineage."
ACTION_ORIGIN_MARKER_KEY = "__projection_origin__"
ACTION_ORIGIN_MARKER_SCHEMA = "projection.action_origin.v1"
ORIGIN_CONTRACT_PREFIX = "ACG_ORIGINS: "
PARSE_STATUSES = frozenset(
    {
        "mapping_tool_calls",
        "mapping_single_call",
        "json_tool_calls",
        "json_single_call",
        "invalid_json",
        "unsupported_payload",
        "observation_mapping",
        "observation_json",
        "observation_unstructured",
    }
)


@dataclass(frozen=True)
class OriginLineage:
    source_id: str
    tool_call_id: str | None
    origin_requirement_ids: tuple[str, ...]
    parent_action_source_id: str | None
    basis: str
    parse_status: str = "mapping_tool_calls"
    marker_status: str = "absent"
    call_id_status: str = "missing"

    def __post_init__(self) -> None:
        if self.basis not in LINEAGE_BASES:
            raise ValueError(f"unsupported lineage basis: {self.basis!r}")
        if self.parse_status not in PARSE_STATUSES:
            raise ValueError(f"unsupported lineage parse status: {self.parse_status!r}")


class RequirementLineageIndex:
    def __init__(self, requirements: Sequence[RequirementAnchor] = ()) -> None:
        self.requirements: tuple[RequirementAnchor, ...] = tuple(requirements)
        self._by_source: dict[str, tuple[OriginLineage, ...]] = {}
        self._by_call_id: dict[str, OriginLineage] = {}
        self._object_origins: dict[str, tuple[str, ...]] = {}
        self._raw_actions: dict[str, Any] = {}
        self._raw_observations: dict[str, tuple[Any, str | None]] = {}
        self._raw_order: list[tuple[str, str]] = []
        self._telemetry: dict[str, dict[str, Any]] = {}
        self._later_user_source_ids: set[str] = set()
        self._strict_origin_action_ids: set[str] = set()
        self._structured_action_origins: dict[
            str, dict[str | None, tuple[str, ...] | None]
        ] = {}
        self._user_source_decision_origins: dict[str, OriginLineage] = {}
        self._user_source_decision_metadata: dict[str, dict[str, Any]] = {}

    def set_requirements(self, requirements: Sequence[RequirementAnchor]) -> None:
        self.requirements = tuple(requirements)
        if self._raw_actions or self._raw_observations:
            self._rebuild_from_raw()

    def restore_snapshot(
        self,
        snapshot: Mapping[str, Any],
        *,
        requirements: Sequence[RequirementAnchor],
        sources: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(snapshot, Mapping):
            raise ValueError("Lineage snapshot must be a mapping")
        if snapshot.get("schema_version") != "projection.lineage.":
            raise ValueError("Unsupported ACG lineage snapshot schema")
        raw_sources = snapshot.get("sources")
        if not isinstance(raw_sources, Mapping):
            raise ValueError("Lineage snapshot is missing sources")
        self.requirements = tuple(requirements)
        self._by_source.clear()
        self._by_call_id.clear()
        self._object_origins.clear()
        self._raw_actions.clear()
        self._raw_observations.clear()
        self._raw_order.clear()
        self._telemetry = {
            str(source_id): dict(value)
            for source_id, value in (snapshot.get("telemetry") or {}).items()
            if isinstance(value, Mapping)
        }
        self._later_user_source_ids = {
            str(source_id)
            for source_id, telemetry in self._telemetry.items()
            if telemetry.get("later_user_source") is True
        }
        self._strict_origin_action_ids.clear()
        self._structured_action_origins.clear()
        self._user_source_decision_origins.clear()
        self._user_source_decision_metadata.clear()
        source_items = list((sources or {}).values())
        source_items.sort(key=lambda item: int(getattr(item, "sequence", 0)))
        for source in source_items:
            source_id = str(getattr(source, "id", ""))
            kind = str(getattr(getattr(source, "kind", None), "value", ""))
            if kind == "action":
                self._raw_actions[source_id] = getattr(source, "payload", None)
                self._raw_order.append(("action", source_id))
            elif kind == "observation":
                parent = self._nearest_action_source(
                    source,
                    {str(getattr(item, "id", "")): item for item in source_items},
                )
                self._raw_observations[source_id] = (
                    getattr(source, "payload", None),
                    parent,
                )
                self._raw_order.append(("observation", source_id))
        for source_id, raw_records in raw_sources.items():
            if not isinstance(raw_records, Sequence) or isinstance(
                raw_records, (str, bytes)
            ):
                raise ValueError(f"Lineage records for {source_id!r} are not an array")
            restored: list[OriginLineage] = []
            for raw in raw_records:
                if not isinstance(raw, Mapping):
                    raise ValueError(
                        f"Lineage record for {source_id!r} is not an object"
                    )
                record = OriginLineage(
                    source_id=str(raw.get("source_id") or source_id),
                    tool_call_id=str(raw["tool_call_id"])
                    if raw.get("tool_call_id") is not None
                    else None,
                    origin_requirement_ids=tuple(
                        (str(item) for item in raw.get("origin_requirement_ids") or ())
                    ),
                    parent_action_source_id=str(raw["parent_action_source_id"])
                    if raw.get("parent_action_source_id") is not None
                    else None,
                    basis=str(raw.get("basis")),
                    parse_status=str(raw.get("parse_status")),
                    marker_status=str(raw.get("marker_status") or "absent"),
                    call_id_status=str(raw.get("call_id_status") or "missing"),
                )
                unknown = set(record.origin_requirement_ids) - {
                    item.id for item in self.requirements
                }
                if unknown:
                    raise ValueError(
                        f"Lineage record {source_id!r} references unknown requirements: {sorted(unknown)}"
                    )
                restored.append(record)
                if record.tool_call_id:
                    self._by_call_id[record.tool_call_id] = record
                for requirement_id in record.origin_requirement_ids:
                    self._object_origins.setdefault(requirement_id, ())
            self._by_source[str(source_id)] = tuple(restored)
        for source_id, action in self._raw_actions.items():
            calls, _status, _marker = _action_calls(action)
            records_by_call = {
                item.tool_call_id: item
                for item in self._by_source.get(source_id, ())
                if item.tool_call_id
            }
            for call in calls:
                record = records_by_call.get(_call_id(call))
                if record is None:
                    continue
                for value in _stable_id_values(_call_arguments(call)):
                    normalized = _normalize_scalar(value)
                    if normalized and record.origin_requirement_ids:
                        existing = self._object_origins.get(normalized, ())
                        self._object_origins[normalized] = tuple(
                            dict.fromkeys((*existing, *record.origin_requirement_ids))
                        )
        for source_id, telemetry in self._telemetry.items():
            decision = telemetry.get("user_source_decision")
            if isinstance(decision, Mapping):
                records = self._by_source.get(source_id, ())
                if records:
                    self._user_source_decision_origins[source_id] = records[0]
                self._user_source_decision_metadata[source_id] = dict(decision)

    @property
    def telemetry(self) -> Mapping[str, Mapping[str, Any]]:
        return {key: dict(value) for key, value in self._telemetry.items()}

    def sync_plugin(self, plugin: Any) -> None:
        sources = getattr(getattr(plugin, "sources", None), "records", {})
        if not isinstance(sources, Mapping):
            return
        self._raw_actions.clear()
        self._raw_observations.clear()
        self._raw_order.clear()
        for source in sorted(
            sources.values(), key=lambda item: getattr(item, "sequence", 0)
        ):
            kind = str(
                getattr(
                    getattr(source, "kind", None), "value", getattr(source, "kind", "")
                )
            )
            source_id = str(getattr(source, "id", ""))
            if kind == "action":
                self._raw_actions[source_id] = getattr(source, "payload", None)
                self._raw_order.append(("action", source_id))
            elif kind == "observation":
                parent = self._nearest_action_source(source, sources)
                self._raw_observations[source_id] = (
                    getattr(source, "payload", None),
                    parent,
                )
                self._raw_order.append(("observation", source_id))
        self._rebuild_from_raw()

    @staticmethod
    def _nearest_action_source(source: Any, sources: Mapping[str, Any]) -> str | None:
        sequence = getattr(source, "sequence", 0)
        actions = [
            item
            for item in sources.values()
            if str(
                getattr(getattr(item, "kind", None), "value", getattr(item, "kind", ""))
            )
            == "action"
            and getattr(item, "sequence", 0) < sequence
        ]
        if not actions:
            return None
        return str(max(actions, key=lambda item: getattr(item, "sequence", 0)).id)

    def _rebuild_from_raw(self) -> None:
        self._by_source.clear()
        self._by_call_id.clear()
        self._object_origins.clear()
        self._telemetry.clear()
        for kind, source_id in self._raw_order:
            if kind == "action":
                self._record_action(
                    source_id,
                    self._raw_actions[source_id],
                    store_raw=False,
                    strict_origin=source_id in self._strict_origin_action_ids,
                )
            else:
                payload, parent = self._raw_observations[source_id]
                if source_id in self._later_user_source_ids:
                    parent = None
                self._inherit_observation(source_id, payload, parent, store_raw=False)
        for source_id, record in self._user_source_decision_origins.items():
            if source_id in self._by_source:
                self._by_source[source_id] = (record,)
            metadata = self._user_source_decision_metadata.get(source_id)
            if metadata is not None:
                telemetry = self._telemetry.setdefault(source_id, {})
                telemetry["later_user_source"] = True
                telemetry["user_source_decision"] = dict(metadata)
                telemetry["unassigned_reason"] = None

    def mark_later_user_source(self, source_id: str) -> None:
        source_id = str(source_id)
        self._later_user_source_ids.add(source_id)
        if source_id in self._raw_observations:
            payload, _parent = self._raw_observations[source_id]
            self._raw_observations[source_id] = (payload, None)
            self._rebuild_from_raw()
        telemetry = self._telemetry.setdefault(source_id, {})
        telemetry["later_user_source"] = True
        telemetry["unassigned_reason"] = "later_user_without_strict_origin"

    def is_later_user_source(self, source_id: str) -> bool:
        return str(source_id) in self._later_user_source_ids

    def record_action_origins(
        self,
        source_id: str,
        action: Any,
        *,
        requirements: Sequence[RequirementAnchor] | None = None,
    ) -> tuple[OriginLineage, ...]:
        if requirements is not None:
            self.set_requirements(requirements)
        source_id = str(source_id)
        if self._later_user_source_ids:
            self._strict_origin_action_ids.add(source_id)
        self._remember_raw("action", source_id, action)
        return self._record_action(
            source_id,
            action,
            store_raw=False,
            strict_origin=source_id in self._strict_origin_action_ids,
        )

    def _record_action(
        self,
        source_id: str,
        action: Any,
        *,
        store_raw: bool,
        strict_origin: bool = False,
    ) -> tuple[OriginLineage, ...]:
        if store_raw:
            self._remember_raw("action", source_id, action)
        calls, parse_status, top_marker = _action_calls(action)
        if not calls:
            calls = ({"id": None, "arguments": action},)
        records: list[OriginLineage] = []
        for index, call in enumerate(calls):
            call_id = _call_id(call)
            arguments = _call_arguments(call)
            marker, marker_status = _origin_marker_for_call(
                top_marker, call, index, call_id
            )
            structured_by_call = self._structured_action_origins.get(str(source_id), {})
            structured_basis = call_id in structured_by_call
            if call_id in structured_by_call:
                structured_ids = structured_by_call[call_id]
                if structured_ids is None:
                    marker, marker_status = (
                        {"schema_version": "projection.action_origin.invalid"},
                        "invalid",
                    )
                else:
                    marker, marker_status = (
                        origin_marker_for_requirements(structured_ids),
                        "valid",
                    )
            marked_ids = _validated_marker_origins(marker, self.requirements)
            if structured_basis and marker_status == "valid":
                origin_ids, basis = (marked_ids, "structured_builder_origin")
            elif marker_status == "invalid":
                origin_ids, basis = ((), "exact_action_argument")
            elif marker is not None:
                origin_ids, basis = (
                    marked_ids,
                    "explicit_action_origin_marker"
                    if marked_ids
                    else "exact_action_argument",
                )
            elif marked_ids:
                origin_ids, basis = (marked_ids, "explicit_action_origin_marker")
            elif strict_origin:
                origin_ids, basis = ((), "exact_action_argument")
            elif marker_status == "absent" and len(self.requirements) > 1:
                origin_ids, basis = ((), "exact_action_argument")
            else:
                origin_ids, basis = self._action_origins(arguments)
            record = OriginLineage(
                source_id=str(source_id),
                tool_call_id=call_id,
                origin_requirement_ids=origin_ids,
                parent_action_source_id=None,
                basis=basis,
                parse_status=parse_status,
                marker_status=marker_status,
                call_id_status="present" if call_id else "missing",
            )
            records.append(record)
            if call_id:
                self._by_call_id[call_id] = record
            for value in _stable_id_values(arguments):
                normalized = _normalize_scalar(value)
                value_origin_ids = self._direct_requirement_ids(value)
                if not value_origin_ids and len(self.requirements) == 1:
                    value_origin_ids = origin_ids
                if not value_origin_ids and origin_ids:
                    value_origin_ids = tuple(self._object_origins.get(normalized, ()))
                if normalized and value_origin_ids:
                    existing = self._object_origins.get(normalized, ())
                    self._object_origins[normalized] = tuple(
                        dict.fromkeys((*existing, *value_origin_ids))
                    )
        self._by_source[str(source_id)] = tuple(records)
        self._telemetry[str(source_id)] = {
            "kind": "action",
            "parse_status": parse_status,
            "marker_status": "present"
            if any((item.marker_status == "valid" for item in records))
            else "invalid"
            if any((item.marker_status == "invalid" for item in records))
            else "absent",
            "call_ids": [item.tool_call_id for item in records if item.tool_call_id],
            "call_id_status": "all_present"
            if records and all((item.tool_call_id for item in records))
            else "missing_or_partial",
            "call_count": len(records),
            "qualified_call_count": sum(
                (bool(item.origin_requirement_ids) for item in records)
            ),
            "origin_requirement_ids": sorted(
                {rid for item in records for rid in item.origin_requirement_ids}
            ),
            "origin_contract_status": "qualified"
            if any((item.origin_requirement_ids for item in records))
            else "unassigned",
            "unassigned_reason": None
            if any((item.origin_requirement_ids for item in records))
            else "no_strict_origin_marker_or_exact_requirement_argument",
        }
        return tuple(records)

    def record_action_contract(
        self, source_id: str, contract: Mapping[str, Any]
    ) -> None:
        telemetry = self._telemetry.setdefault(str(source_id), {})
        telemetry["producer_contract"] = dict(contract)
        if not telemetry.get("origin_requirement_ids"):
            telemetry["origin_contract_status"] = "unassigned"
            telemetry["unassigned_reason"] = str(
                contract.get("unassigned_reason") or "origin_not_provided"
            )

    def record_user_source_decision(
        self,
        source_id: str,
        *,
        decision: str,
        exact_spans: Sequence[str],
        continuation_requirement_ids: Sequence[str],
        appended_requirement_ids: Sequence[str] = (),
        segments: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        source_id = str(source_id)
        requirement_ids = tuple(
            dict.fromkeys(
                (
                    str(item)
                    for item in (
                        *continuation_requirement_ids,
                        *appended_requirement_ids,
                    )
                )
            )
        )
        origin = OriginLineage(
            source_id=source_id,
            tool_call_id=None,
            origin_requirement_ids=requirement_ids,
            parent_action_source_id=None,
            basis="user_source_decision",
            parse_status="observation_unstructured",
            marker_status="explicit",
            call_id_status="not_applicable",
        )
        self._user_source_decision_origins[source_id] = origin
        self._by_source[source_id] = (origin,)
        telemetry = self._telemetry.setdefault(str(source_id), {})
        metadata = {
            "status": "resolved",
            "decision": str(decision),
            "exact_spans": [str(item) for item in exact_spans],
            "continuation_requirement_ids": [
                str(item) for item in continuation_requirement_ids
            ],
            "appended_requirement_ids": [
                str(item) for item in appended_requirement_ids
            ],
            "segments": [dict(item) for item in segments],
            "origin_requirement_ids": list(requirement_ids),
        }
        self._user_source_decision_metadata[source_id] = dict(metadata)
        telemetry["later_user_source"] = True
        telemetry["user_source_decision"] = metadata
        telemetry["unassigned_reason"] = None

    def record_structured_action_origins(
        self, source_id: str, payload: Any
    ) -> dict[str, Any]:
        source_id = str(source_id)
        entries = _structured_action_origins(payload)
        calls, _parse_status, _marker = _action_calls(self._raw_actions.get(source_id))
        call_ids = tuple((_call_id(call) for call in calls))
        known_calls = set(call_ids)
        known_requirements = {item.id for item in self.requirements}
        overrides: dict[str, tuple[str, ...] | None] = {}
        issues: list[str] = []
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            entries = ()
            issues.append("action_origins_not_array")
        if len(entries) != len(call_ids) or any(
            (call_id is None for call_id in call_ids)
        ):
            issues.append("action_origins_call_coverage_mismatch")
        seen_calls: set[str] = set()
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                issues.append(f"action_origins[{index}]_not_mapping")
                continue
            entry_source_id = _structured_action_source_id(entry, source_id)
            if entry_source_id != source_id:
                issues.append(f"action_origins[{index}]_source_id_mismatch")
                continue
            call_id = entry.get("tool_call_id")
            requirement_ids = entry.get("requirement_ids")
            if not isinstance(call_id, str) or call_id not in known_calls:
                issues.append(f"action_origins[{index}]_unknown_tool_call_id")
                continue
            if call_id in seen_calls:
                issues.append(f"action_origins[{index}]_duplicate_tool_call_id")
                continue
            seen_calls.add(call_id)
            if not isinstance(requirement_ids, Sequence) or isinstance(
                requirement_ids, (str, bytes)
            ):
                overrides[call_id] = None
                issues.append(f"action_origins[{index}]_requirement_ids_not_array")
                continue
            values = tuple(requirement_ids)
            if any(
                (
                    not isinstance(item, str) or item not in known_requirements
                    for item in values
                )
            ):
                overrides[call_id] = None
                issues.append(f"action_origins[{index}]_unknown_requirement_id")
                continue
            if len(set(values)) != len(values):
                overrides[call_id] = None
                issues.append(f"action_origins[{index}]_duplicate_requirement_id")
                continue
            overrides[call_id] = values
        if set(seen_calls) != known_calls:
            issues.append("action_origins_call_coverage_incomplete")
        if issues:
            overrides = {call_id: None for call_id in known_calls}
        else:
            overrides = {call_id: overrides[call_id] for call_id in known_calls}
        self._structured_action_origins[source_id] = overrides
        if source_id in self._raw_actions:
            self._rebuild_from_raw()
        telemetry = self._telemetry.setdefault(source_id, {})
        telemetry["structured_action_origins"] = {
            "status": "valid" if not issues else "invalid",
            "issues": list(issues),
            "entry_count": len(entries),
        }
        return telemetry["structured_action_origins"]

    def inherit_observation_origins(
        self, source_id: str, observation: Any, *, parent_action_source_id: str | None
    ) -> tuple[OriginLineage, ...]:
        source_id = str(source_id)
        self._remember_raw(
            "observation", source_id, (observation, parent_action_source_id)
        )
        return self._inherit_observation(
            source_id, observation, parent_action_source_id, store_raw=False
        )

    def _inherit_observation(
        self,
        source_id: str,
        observation: Any,
        parent_action_source_id: str | None,
        *,
        store_raw: bool,
    ) -> tuple[OriginLineage, ...]:
        if str(source_id) in self._later_user_source_ids:
            parent_action_source_id = None
        if store_raw:
            self._remember_raw(
                "observation", source_id, (observation, parent_action_source_id)
            )
        action_records = self._by_source.get(str(parent_action_source_id), ())
        observation_call_ids = _observation_call_ids(observation)
        parse_status = _observation_parse_status(observation)
        records: list[OriginLineage] = []
        if observation_call_ids:
            for call_id in observation_call_ids:
                parent = self._by_call_id.get(call_id)
                if parent is None:
                    continue
                records.append(
                    OriginLineage(
                        source_id=str(source_id),
                        tool_call_id=call_id,
                        origin_requirement_ids=parent.origin_requirement_ids,
                        parent_action_source_id=parent_action_source_id,
                        basis="tool_observation_inheritance",
                        parse_status=parse_status,
                        marker_status="inherited",
                        call_id_status="matched",
                    )
                )
        elif (
            len(action_records) == 1
            and str(source_id) not in self._later_user_source_ids
        ):
            parent = action_records[0]
            records.append(
                OriginLineage(
                    source_id=str(source_id),
                    tool_call_id=parent.tool_call_id,
                    origin_requirement_ids=parent.origin_requirement_ids,
                    parent_action_source_id=parent_action_source_id,
                    basis="tool_observation_inheritance",
                    parse_status=parse_status,
                    marker_status="inherited",
                    call_id_status="inherited_single_action",
                )
            )
        self._by_source[str(source_id)] = tuple(records)
        self._telemetry[str(source_id)] = {
            "kind": "observation",
            "parse_status": parse_status,
            "marker_status": "inherited",
            "call_ids": list(observation_call_ids),
            "call_id_status": "matched"
            if observation_call_ids and records
            else "inherited_single_action"
            if records
            else "unknown_batch_or_unmatched",
            "call_count": len(observation_call_ids),
            "qualified_call_count": sum(
                (bool(item.origin_requirement_ids) for item in records)
            ),
            "origin_requirement_ids": sorted(
                {rid for item in records for rid in item.origin_requirement_ids}
            ),
            "parent_action_source_id": parent_action_source_id,
            "later_user_source": str(source_id) in self._later_user_source_ids,
            "unassigned_reason": None
            if records and any((item.origin_requirement_ids for item in records))
            else "later_user_without_strict_origin"
            if str(source_id) in self._later_user_source_ids
            else "unknown_batch_or_unmatched",
        }
        return tuple(records)

    def _remember_raw(self, kind: str, source_id: str, payload: Any) -> None:
        source_id = str(source_id)
        if kind == "action":
            self._raw_actions[source_id] = payload
        else:
            self._raw_observations[source_id] = payload
        if (kind, source_id) not in self._raw_order:
            self._raw_order.append((kind, source_id))

    def origins_for_source(self, source_id: str) -> tuple[OriginLineage, ...]:
        return self._by_source.get(str(source_id), ())

    def origins_for_memory_event(
        self, memory: Any, event: Any | None
    ) -> tuple[OriginLineage, ...]:
        source_ids = tuple(
            (str(item) for item in getattr(memory, "source_refs", ()) or ())
        )
        records = tuple(
            (
                record
                for source_id in source_ids
                for record in self._by_source.get(source_id, ())
            )
        )
        if not records:
            return ()
        event_evidence = (
            set(getattr(event, "evidence_ids", ()) or ()) if event else set()
        )
        if event is not None and getattr(memory, "id", None) not in event_evidence:
            return ()
        call_ids = set(_observation_call_ids(getattr(memory, "content", "")))
        if call_ids:
            return tuple(
                (record for record in records if record.tool_call_id in call_ids)
            )
        return records if len(records) == 1 else ()

    def _action_origins(self, arguments: Any) -> tuple[tuple[str, ...], str]:
        if len(self.requirements) == 1:
            return ((self.requirements[0].id,), "single_requirement_task")
        direct: list[str] = []
        for requirement in self.requirements:
            if any(
                (
                    _exact_scalar_in_text(value, requirement.exact_text)
                    for value in _stable_id_values(arguments)
                )
            ):
                direct.append(requirement.id)
        if direct:
            return (tuple(dict.fromkeys(direct)), "exact_action_argument")
        inherited: list[str] = []
        for value in _stable_id_values(arguments):
            inherited.extend(self._object_origins.get(_normalize_scalar(value), ()))
        if inherited:
            return (tuple(dict.fromkeys(inherited)), "inherited_object_origin")
        return ((), "exact_action_argument")

    def _direct_requirement_ids(self, value: str) -> tuple[str, ...]:
        return tuple(
            (
                requirement.id
                for requirement in self.requirements
                if _exact_scalar_in_text(value, requirement.exact_text)
            )
        )


def install_lineage_record_hook(
    plugin: Any, index: RequirementLineageIndex | None = None
) -> RequirementLineageIndex:
    existing = getattr(plugin, "lineage", None)
    if existing is not None and getattr(plugin, "_lineage_hooked", False):
        return existing
    lineage = index or RequirementLineageIndex()
    original_record = plugin.record
    last_action_source_id: str | None = None

    def record(kind: Any, payload: Any) -> Any:
        nonlocal last_action_source_id
        source = original_record(kind, payload)
        normalized_kind = getattr(kind, "value", kind)
        if str(normalized_kind) == "action":
            contract = action_origin_contract(payload)
            lineage.record_action_origins(source.id, payload)
            lineage.record_action_contract(source.id, contract)
            last_action_source_id = source.id
        elif str(normalized_kind) == "observation":
            lineage.inherit_observation_origins(
                source.id,
                payload,
                parent_action_source_id=None
                if lineage.is_later_user_source(source.id)
                else last_action_source_id,
            )
        return source

    plugin.record = record
    plugin.lineage = lineage
    plugin._lineage_hooked = True
    original_update = plugin.update

    def update_with_structured_origins(source_ids: Sequence[str]) -> Any:
        result = original_update(source_ids)
        raw_output = getattr(
            getattr(result, "concept_proposal", None), "raw_output", None
        )
        concept_indexer = getattr(plugin, "concept_indexer", None)
        apply_dynamic = getattr(concept_indexer, "apply_dynamic_requirements", None)
        if callable(apply_dynamic):
            apply_dynamic(raw_output)
        entries = _structured_action_origins(raw_output)
        if entries is not None:
            for source_id in getattr(result, "source_ids", ()):
                if str(source_id) in lineage._raw_actions:
                    lineage.record_structured_action_origins(source_id, raw_output)
        return result

    plugin.update = update_with_structured_origins
    plugin._lineage_update_hooked = True
    lineage.sync_plugin(plugin)
    sources = getattr(getattr(plugin, "sources", None), "records", {})
    actions = (
        [
            source
            for source in sources.values()
            if str(
                getattr(
                    getattr(source, "kind", None), "value", getattr(source, "kind", "")
                )
            )
            == "action"
        ]
        if isinstance(sources, Mapping)
        else []
    )
    if actions:
        last_action_source_id = str(
            max(actions, key=lambda item: getattr(item, "sequence", 0)).id
        )
    return lineage


def action_origin_contract(action: Any) -> dict[str, Any]:
    calls, parse_status, top_marker = _action_calls(action)
    if not calls:
        calls = ({"id": None, "arguments": action},)
    statuses: list[str] = []
    call_ids: list[str] = []
    for index, call in enumerate(calls):
        call_id = _call_id(call)
        if call_id:
            call_ids.append(call_id)
        _, status = _origin_marker_for_call(top_marker, call, index, call_id)
        statuses.append(status)
    marker_status = (
        "valid"
        if any((item == "valid" for item in statuses))
        else "invalid"
        if any((item == "invalid" for item in statuses))
        else "absent"
    )
    return {
        "schema_version": ACTION_ORIGIN_MARKER_SCHEMA,
        "parse_status": parse_status,
        "marker_status": marker_status,
        "call_ids": call_ids,
        "call_id_status": "all_present"
        if calls and len(call_ids) == len(calls)
        else "missing_or_partial",
        "status": "provided" if marker_status == "valid" else "unassigned",
        "unassigned_reason": None
        if marker_status == "valid"
        else "origin_marker_absent"
        if marker_status == "absent"
        else "origin_marker_invalid",
    }


def origin_contract_instruction(requirement_ids: Sequence[str]) -> str:
    ids = tuple((str(item) for item in requirement_ids))
    rendered_ids = ", ".join(ids) if ids else "(none)"
    return f'ACG origin contract (do not put this in tool arguments or tool schemas): when returning one or more native tool calls, emit the single assistant content line `ACG_ORIGINS: [["R1"], ["R2"]]`. The outer list must have exactly one entry per tool call, in native tool-call order; use [] when the origin is unknown. Only these current source-grounded requirement IDs are valid: {rendered_ids}. Do not add any other content on that line.'


def parse_origin_contract_content(
    content: Any, *, call_count: int, requirement_ids: Sequence[str]
) -> tuple[tuple[tuple[str, ...], ...], str]:
    if content in (None, ""):
        return (tuple((() for _ in range(max(call_count, 0)))), "absent")
    if not isinstance(content, str) or len(content.splitlines()) != 1:
        return (tuple((() for _ in range(max(call_count, 0)))), "invalid")
    if not content.startswith(ORIGIN_CONTRACT_PREFIX):
        return (tuple((() for _ in range(max(call_count, 0)))), "absent")
    raw = content[len(ORIGIN_CONTRACT_PREFIX) :]
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return (tuple((() for _ in range(max(call_count, 0)))), "invalid")
    if not isinstance(parsed, list) or len(parsed) != call_count:
        return (tuple((() for _ in range(max(call_count, 0)))), "invalid")
    known = {str(item) for item in requirement_ids}
    declarations: list[tuple[str, ...]] = []
    for item in parsed:
        if not isinstance(item, list):
            return (tuple((() for _ in range(call_count))), "invalid")
        if any((not isinstance(value, str) or value not in known for value in item)):
            return (tuple((() for _ in range(call_count))), "invalid")
        if len(set(item)) != len(item):
            return (tuple((() for _ in range(call_count))), "invalid")
        declarations.append(tuple(item))
    return (tuple(declarations), "valid")


def origin_marker_for_requirements(requirement_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": ACTION_ORIGIN_MARKER_SCHEMA,
        "requirement_ids": list(dict.fromkeys((str(item) for item in requirement_ids))),
    }


def attach_origin_marker(
    action: Any, requirement_ids: Sequence[str], *, call_index: int = 0
) -> Any:
    marker = origin_marker_for_requirements(requirement_ids)
    if isinstance(action, Mapping):
        envelope = dict(action)
        calls = envelope.get("tool_calls")
        if isinstance(calls, Sequence) and (not isinstance(calls, (str, bytes))):
            copied_calls = [
                dict(item) if isinstance(item, Mapping) else item for item in calls
            ]
            if copied_calls and 0 <= call_index < len(copied_calls):
                copied_calls[call_index][ACTION_ORIGIN_MARKER_KEY] = marker
                envelope["tool_calls"] = [copied_calls[call_index]]
            else:
                envelope[ACTION_ORIGIN_MARKER_KEY] = marker
            return envelope
        envelope[ACTION_ORIGIN_MARKER_KEY] = marker
        return envelope
    if isinstance(action, str):
        try:
            parsed = json.loads(action)
        except (TypeError, json.JSONDecodeError):
            return {"arguments": action, ACTION_ORIGIN_MARKER_KEY: marker}
        return attach_origin_marker(parsed, requirement_ids, call_index=call_index)
    return {"arguments": action, ACTION_ORIGIN_MARKER_KEY: marker}


def _action_calls(
    value: Any,
) -> tuple[tuple[Mapping[str, Any], ...], str, Mapping[str, Any] | None]:
    if isinstance(value, Mapping):
        top_marker = value.get(ACTION_ORIGIN_MARKER_KEY)
        calls = value.get("tool_calls")
        if isinstance(calls, Sequence) and (not isinstance(calls, (str, bytes))):
            return (
                tuple((item for item in calls if isinstance(item, Mapping))),
                "mapping_tool_calls",
                top_marker if isinstance(top_marker, Mapping) else None,
            )
        if "function" in value and isinstance(value["function"], Mapping):
            return (
                (value,),
                "mapping_single_call",
                top_marker if isinstance(top_marker, Mapping) else None,
            )
        if any((key in value for key in ("tool", "name", "arguments", "args"))):
            return (
                (value,),
                "mapping_single_call",
                top_marker if isinstance(top_marker, Mapping) else None,
            )
        return (
            (),
            "unsupported_payload",
            top_marker if isinstance(top_marker, Mapping) else None,
        )
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return ((), "invalid_json", None)
        calls, _, marker = _action_calls(parsed)
        return (
            calls,
            "json_tool_calls" if calls and len(calls) > 1 else "json_single_call",
            marker,
        )
    return ((), "unsupported_payload", None)


def _structured_action_origins(value: Any) -> Any:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return None
    if isinstance(value, Mapping):
        return value.get("action_origins")
    return None


def _structured_action_source_id(
    entry: Mapping[str, Any], source_id: str
) -> str | None:
    value = entry.get("action_source_id")
    return source_id if isinstance(value, str) and value == source_id else None


def _tool_calls(value: Any) -> tuple[Mapping[str, Any], ...]:
    return _action_calls(value)[0]


def _origin_marker_for_call(
    top_marker: Mapping[str, Any] | None,
    call: Mapping[str, Any],
    index: int,
    call_id: str | None,
) -> tuple[Mapping[str, Any] | None, str]:
    marker = call.get(ACTION_ORIGIN_MARKER_KEY)
    if isinstance(marker, Mapping):
        return (marker, "valid" if _marker_shape_is_valid(marker) else "invalid")
    if not isinstance(top_marker, Mapping):
        return (None, "absent")
    call_markers = top_marker.get("calls")
    if isinstance(call_markers, Sequence) and (
        not isinstance(call_markers, (str, bytes))
    ):
        for item in call_markers:
            if not isinstance(item, Mapping):
                continue
            item_call_id = item.get("tool_call_id", item.get("call_id", item.get("id")))
            if (
                call_id
                and str(item_call_id) == call_id
                or (call_id is None and index == call_markers.index(item))
            ):
                return (item, "valid" if _marker_shape_is_valid(item) else "invalid")
        return (None, "invalid")
    return (top_marker, "valid" if _marker_shape_is_valid(top_marker) else "invalid")


def _marker_shape_is_valid(marker: Mapping[str, Any]) -> bool:
    return (
        marker.get("schema_version") == ACTION_ORIGIN_MARKER_SCHEMA
        and isinstance(marker.get("requirement_ids"), Sequence)
        and (not isinstance(marker.get("requirement_ids"), (str, bytes)))
    )


def _validated_marker_origins(
    marker: Mapping[str, Any] | None, requirements: Sequence[RequirementAnchor]
) -> tuple[str, ...]:
    if marker is None or not _marker_shape_is_valid(marker):
        return ()
    known = {item.id for item in requirements}
    values = marker.get("requirement_ids") or ()
    if not values or any(
        (not isinstance(item, str) or item not in known for item in values)
    ):
        return ()
    return tuple(dict.fromkeys(values))


def _observation_parse_status(value: Any) -> str:
    if isinstance(value, Mapping):
        return "observation_mapping"
    if isinstance(value, str):
        try:
            json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return "observation_unstructured"
        return "observation_json"
    return "observation_unstructured"


def _call_id(call: Mapping[str, Any]) -> str | None:
    value = call.get("tool_call_id", call.get("id"))
    return str(value) if value not in (None, "") else None


def _call_arguments(call: Mapping[str, Any]) -> Any:
    arguments = call.get("arguments")
    function = call.get("function")
    if arguments is None and isinstance(function, Mapping):
        arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return arguments if arguments is not None else call


def _observation_call_ids(value: Any) -> tuple[str, ...]:
    found: list[str] = []
    if isinstance(value, Mapping):
        direct = value.get("tool_call_id")
        if direct not in (None, ""):
            found.append(str(direct))
        calls = value.get("tool_calls")
        if isinstance(calls, Sequence) and (not isinstance(calls, (str, bytes))):
            for item in calls:
                found.extend(_observation_call_ids(item))
    elif isinstance(value, str):
        for match in re.finditer(
            "(?:tool_call_id|call_id)\\s*['\\\"]?\\s*[:=]\\s*['\\\"]([^'\\\",}\\s]+)",
            value,
        ):
            found.append(match.group(1))
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            parsed = None
        if parsed is not None:
            found.extend(_observation_call_ids(parsed))
    return tuple(dict.fromkeys(found))


def _stable_id_values(value: Any, *, key_hint: str = "") -> tuple[str, ...]:
    if isinstance(value, Mapping):
        return tuple(
            (
                item
                for key, child in value.items()
                for item in _stable_id_values(child, key_hint=str(key))
            )
        )
    if isinstance(value, Sequence) and (not isinstance(value, (str, bytes))):
        return tuple(
            (
                item
                for child in value
                for item in _stable_id_values(child, key_hint=key_hint)
            )
        )
    if not _is_id_like_key(key_hint) or not isinstance(value, str):
        return ()
    normalized = _normalize_scalar(value)
    return (normalized,) if normalized else ()


def _is_id_like_key(key: str) -> bool:
    normalized = str(key).strip().casefold().replace("-", "_")
    terminal = normalized.rsplit(".", 1)[-1]
    return (
        terminal
        in {"id", "ids", "uuid", "uuids", "key", "keys", "identifier", "identifiers"}
        or terminal.endswith("_id")
        or terminal.endswith("_uuid")
        or terminal.endswith("_key")
        or terminal.endswith("_identifier")
    )


def _normalize_scalar(value: Any) -> str:
    return str(value).strip().casefold()


def _exact_scalar_in_text(value: Any, text: str) -> bool:
    normalized = _normalize_scalar(value)
    if not normalized:
        return False
    return (
        re.search(
            f"(?<![A-Za-z0-9]){re.escape(normalized)}(?![A-Za-z0-9])",
            str(text).casefold(),
        )
        is not None
    )


__all__ = [
    "ACTION_ORIGIN_MARKER_KEY",
    "ACTION_ORIGIN_MARKER_SCHEMA",
    "ORIGIN_CONTRACT_PREFIX",
    "LINEAGE_BASES",
    "LINEAGE_SCHEMA_VERSION",
    "OriginLineage",
    "PARSE_STATUSES",
    "RequirementLineageIndex",
    "action_origin_contract",
    "attach_origin_marker",
    "install_lineage_record_hook",
    "origin_contract_instruction",
    "origin_marker_for_requirements",
    "parse_origin_contract_content",
]
