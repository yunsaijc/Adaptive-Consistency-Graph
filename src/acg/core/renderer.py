from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Mapping

from .history import EventStageHistoryProvider, FlatHistoryProvider, HistoryProvider
from .projection import (
    TEMPORAL_CURRENT,
    TEMPORAL_EARLIER,
    TEMPORAL_LATEST,
    TEMPORAL_UNSCOPED,
    Projection,
    SupportIncidence,
)
from .view import approximate_token_count

RENDERER_VERSION = "renderer.v1"
PACKED_INDEX_RENDERER_VERSION = "renderer.packed_index.v1"


class RenderBudgetError(ValueError):
    pass


TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class EvidenceMembership:
    requirement_id: str | None
    object_scope_id: str | None
    event_id: str | None
    temporal_stratum: str
    memory_id: str
    route: str
    rendered_as_evidence: bool
    cross_reference: bool
    rendered_as_membership_shell: bool
    omitted: bool
    omission_reason: str | None
    overlap_requirement_ids: tuple[str, ...]
    binding_basis: str | None = None
    lineage_source_ids: tuple[str, ...] = ()
    lineage_tool_call_ids: tuple[str, ...] = ()

    @property
    def visible(self) -> bool:
        return (
            self.rendered_as_evidence
            or self.cross_reference
            or self.rendered_as_membership_shell
        )


@dataclass(frozen=True)
class TextOmission:
    node_id: str
    source_id: str
    original_start_offset: int
    original_end_offset: int
    rendered_start_offset: int
    rendered_end_offset: int
    content_sha256: str
    reason: str


@dataclass(frozen=True)
class RenderedContext:
    text: str
    selected_memory_ids: tuple[str, ...]
    omitted_memory_ids: tuple[str, ...]
    omission_reasons: Mapping[str, str]
    projection: Projection
    memberships: tuple[EvidenceMembership, ...] = ()
    candidate_order: tuple[str, ...] = ()
    provider: str = "event_stage"
    text_sha256: str = ""
    token_count: int = 0
    current_event_memory_ids: tuple[str, ...] = ()
    current_event_visible_memory_ids: tuple[str, ...] = ()
    text_omissions: tuple[TextOmission, ...] = ()
    renderer_version: str = RENDERER_VERSION
    lineage_telemetry: Mapping[str, Mapping[str, object]] | None = None

    @property
    def selected(self) -> tuple[str, ...]:
        return self.selected_memory_ids

    @property
    def omitted(self) -> tuple[str, ...]:
        return self.omitted_memory_ids

    @property
    def membership(self) -> tuple[EvidenceMembership, ...]:
        return self.memberships

    def __iter__(self) -> Iterator[object]:
        yield self.text
        yield self.selected_memory_ids

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> object:
        return (self.text, self.selected_memory_ids)[index]


def fit_projection(
    projection: Projection,
    *,
    omitted_memory_ids: Sequence[str],
    omission_reasons: Mapping[str, str] | None = None,
) -> Projection:
    available = set(_candidate_memory_ids(projection))
    omitted = tuple(
        (
            memory_id
            for memory_id in dict.fromkeys(omitted_memory_ids)
            if memory_id in available
        )
    )
    reasons = {
        memory_id: str((omission_reasons or {}).get(memory_id, "omitted_by_budget"))
        for memory_id in omitted
    }
    return replace(projection, omitted_memory_ids=omitted, omission_reasons=reasons)


def render_projection(
    projection: Projection,
    *,
    max_tokens: int,
    token_counter: TokenCounter | None = None,
    history_provider: HistoryProvider | None = None,
    memory_line_max_chars: int = 220,
) -> RenderedContext:
    candidate_ids = _candidate_memory_ids(projection)
    current_event_ids = tuple(
        (
            memory_id
            for memory_id in projection.current_event_memory_ids
            if memory_id in candidate_ids
        )
    )
    count = token_counter or approximate_token_count
    provider = history_provider or EventStageHistoryProvider()
    if max_tokens <= 0:
        if projection.incidences:
            raise RenderBudgetError(
                "mandatory requirement membership shell exceeds zero-token budget"
            )
        fitted = fit_projection(
            projection,
            omitted_memory_ids=candidate_ids,
            omission_reasons={
                memory_id: "omitted_by_budget" for memory_id in candidate_ids
            },
        )
        return RenderedContext(
            text="",
            selected_memory_ids=(),
            omitted_memory_ids=fitted.omitted_memory_ids,
            omission_reasons=fitted.omission_reasons or {},
            projection=fitted,
            memberships=_build_memberships(projection, rendered_modes={}),
            candidate_order=candidate_ids,
            provider=provider.name,
            text_sha256=sha256(b"").hexdigest(),
            token_count=0,
            current_event_memory_ids=current_event_ids,
            current_event_visible_memory_ids=(),
            text_omissions=_all_source_text_omissions(
                projection, reason="omitted_by_budget"
            ),
            renderer_version=RENDERER_VERSION,
            lineage_telemetry=getattr(projection, "lineage_telemetry", None),
        )
    task_text, task_omission = _shorten_source_text(
        projection.task_text,
        max_chars=memory_line_max_chars,
        node_id=f"task:{projection.task_source_id}",
        source_id=projection.task_source_id,
        original_start_offset=0,
    )
    text_omissions: list[TextOmission] = []
    if task_omission is not None:
        text_omissions.append(task_omission)
    base_lines = [
        "[ACG task-state projection]",
        "Role: source-constrained hierarchical task-state memory; support links organize records and do not decide completion, conflict, or action.",
        "",
        "[Task source]",
        f"- [{projection.task_source_id}] {task_text}",
        "",
        "[Requirement hierarchy]",
    ]
    for requirement in projection.requirements:
        rendered_requirement, omission = _shorten_source_text(
            requirement.exact_text,
            max_chars=memory_line_max_chars,
            node_id=f"requirement:{requirement.id}",
            source_id=requirement.source_id,
            original_start_offset=requirement.start_offset,
        )
        base_lines.append(f"- {requirement.id}: {rendered_requirement}")
        if omission is not None:
            text_omissions.append(omission)
    queues = _requirement_queues(projection, provider)
    candidate_pairs = _round_robin_pairs(queues)
    candidate_order = tuple(
        dict.fromkeys(
            (
                *(memory_id for _, memory_id in candidate_pairs),
                *projection.current_event_memory_ids,
                *projection.unassigned_current_memory_ids,
            )
        )
    )
    sections: dict[str, list[str]] = {
        requirement.id: ["", f"[Requirement {requirement.id} support]"]
        for requirement in projection.requirements
    }
    current_event_section: list[str] = ["", "[Current event]"]
    shared_section: list[str] = ["", "[Unassigned candidate evidence]"]
    selected: list[str] = []
    omitted: list[str] = []
    rendered_memory_ids: set[str] = set()
    rendered_owner_by_memory: dict[str, str] = {}
    rendered_modes: dict[
        tuple[str | None, str | None, str | None, str, str, str], str
    ] = {}
    for requirement in projection.requirements:
        if not queues.get(requirement.id):
            sections[requirement.id].append(
                "- no source-grounded support record is visible in this projection"
            )
        for incidence in projection.incidences_for_requirement(requirement.id):
            sections[requirement.id].extend(
                _membership_shell_lines(projection, incidence)
            )
            rendered_modes[_incidence_branch_key(incidence)] = "membership_shell"

    def rendered_text() -> str:
        lines = [*base_lines, "", f"[History provider: {provider.name}]"]
        for requirement in projection.requirements:
            lines.extend(sections[requirement.id])
        if projection.current_event_id is not None or current_event_ids:
            lines.extend(current_event_section)
        if projection.shared_candidate_memory_ids:
            lines.extend(shared_section)
        return "\n".join(lines)

    for requirement_id, memory_id in candidate_pairs:
        incidences = _incidences_for_memory(
            projection.incidences_for_requirement(requirement_id), memory_id
        )
        if not incidences:
            continue
        if memory_id in rendered_memory_ids:
            owner_requirement_id = rendered_owner_by_memory[memory_id]
            branch_lines = _cross_reference_lines(
                projection,
                incidences,
                memory_id=memory_id,
                owner_requirement_id=owner_requirement_id,
                overlap_requirement_ids=_overlap_requirement_ids(projection, memory_id),
            )
            sections[requirement_id].extend(branch_lines)
            if count(rendered_text()) <= max_tokens:
                for incidence in incidences:
                    rendered_modes[_incidence_branch_key(incidence)] = "cross_reference"
            else:
                del sections[requirement_id][-len(branch_lines) :]
            continue
        block = _evidence_lines(
            projection, incidences, memory_line_max_chars, provider_name=provider.name
        )
        sections[requirement_id].extend(block)
        if count(rendered_text()) <= max_tokens:
            selected.append(memory_id)
            rendered_memory_ids.add(memory_id)
            rendered_owner_by_memory[memory_id] = requirement_id
            for index, incidence in enumerate(incidences):
                rendered_modes[_incidence_branch_key(incidence)] = (
                    "evidence" if index == 0 else "cross_reference"
                )
        else:
            del sections[requirement_id][-len(block) :]
            omitted.append(memory_id)
    current_event_visible: list[str] = [
        memory_id for memory_id in current_event_ids if memory_id in rendered_memory_ids
    ]
    for memory_id in current_event_ids:
        current_branch_keys = _current_branch_keys(projection, memory_id)
        if memory_id in rendered_memory_ids:
            owner_requirement_id = rendered_owner_by_memory[memory_id]
            line = _current_event_cross_reference_line(
                projection,
                memory_id,
                owner_requirement_id=owner_requirement_id,
                provider_name=provider.name,
            )
            current_event_section.extend(line)
            if count(rendered_text()) <= max_tokens:
                for branch_key in current_branch_keys:
                    rendered_modes[branch_key] = "cross_reference"
                if memory_id not in current_event_visible:
                    current_event_visible.append(memory_id)
            else:
                del current_event_section[-len(line) :]
            continue
        block = _unassigned_lines(
            projection, memory_id, memory_line_max_chars, provider_name=provider.name
        )
        current_event_section.extend(block)
        if count(rendered_text()) <= max_tokens:
            selected.append(memory_id)
            rendered_memory_ids.add(memory_id)
            current_event_visible.append(memory_id)
            for index, branch_key in enumerate(current_branch_keys):
                rendered_modes[branch_key] = (
                    "evidence" if index == 0 else "cross_reference"
                )
        else:
            del current_event_section[-len(block) :]
            omitted.append(memory_id)
    for memory_id in projection.shared_candidate_memory_ids:
        if memory_id not in candidate_ids or memory_id in rendered_memory_ids:
            continue
        block = _shared_candidate_lines(projection, memory_id, memory_line_max_chars)
        shared_section.extend(block)
        if count(rendered_text()) <= max_tokens:
            selected.append(memory_id)
            rendered_memory_ids.add(memory_id)
        else:
            del shared_section[-len(block) :]
            omitted.append(memory_id)
    selected = [memory_id for memory_id in candidate_ids if memory_id in set(selected)]
    omitted = [memory_id for memory_id in candidate_ids if memory_id not in selected]
    lines = rendered_text().splitlines()
    if omitted:
        lines.extend(("", "[Omitted evidence]"))
        for memory_id in omitted:
            lines.append(
                f"- {memory_id}: omitted_by_budget; source refs remain in the graph"
            )
    text = "\n".join(lines)
    result_renderer_version = RENDERER_VERSION
    if count(text) > max_tokens:
        text, selected, current_event_visible, rendered_modes = _render_budget_fallback(
            projection,
            candidate_ids=candidate_ids,
            current_event_ids=current_event_ids,
            provider_name=provider.name,
            max_tokens=max_tokens,
            token_counter=count,
            memory_line_max_chars=memory_line_max_chars,
        )
        selected = [
            memory_id for memory_id in candidate_ids if memory_id in set(selected)
        ]
        omitted = [
            memory_id for memory_id in candidate_ids if memory_id not in selected
        ]
        text_omissions = list(
            _all_source_text_omissions(projection, reason="omitted_by_budget_fallback")
        )
        if text.startswith("[ACG overflow index"):
            result_renderer_version = "renderer.overflow_index.v1"
        elif text.startswith("[ACG packed index"):
            result_renderer_version = PACKED_INDEX_RENDERER_VERSION
    fitted = fit_projection(
        projection,
        omitted_memory_ids=omitted,
        omission_reasons={memory_id: "omitted_by_budget" for memory_id in omitted},
    )
    return RenderedContext(
        text=text.strip(),
        selected_memory_ids=tuple(selected),
        omitted_memory_ids=fitted.omitted_memory_ids,
        omission_reasons=fitted.omission_reasons or {},
        projection=fitted,
        memberships=_build_memberships(projection, rendered_modes=rendered_modes),
        candidate_order=candidate_order,
        provider=provider.name,
        text_sha256=sha256(text.strip().encode("utf-8")).hexdigest(),
        token_count=count(text.strip()),
        current_event_memory_ids=current_event_ids,
        current_event_visible_memory_ids=tuple(current_event_visible),
        text_omissions=tuple(text_omissions),
        renderer_version=result_renderer_version,
        lineage_telemetry=getattr(projection, "lineage_telemetry", None),
    )


def _evidence_lines(
    projection: Projection,
    incidences: Sequence[SupportIncidence],
    max_chars: int,
    *,
    provider_name: str = "event_stage",
) -> tuple[str, ...]:
    if not incidences:
        return ()
    lines: list[str] = []
    for index, incidence in enumerate(incidences):
        object_label = _object_label(projection, incidence.object_scope_id)
        if provider_name == "flat":
            if index:
                lines.append(
                    f"- cross-reference evidence [{incidence.memory_id}] object={object_label} event={incidence.event_id or 'none'} stratum={incidence.temporal_stratum}"
                )
                continue
            lines.extend(
                (
                    f"- evidence [{incidence.memory_id}] object={object_label} event={incidence.event_id or 'none'} stratum={incidence.temporal_stratum} route={incidence.route_basis} binding_basis={incidence.binding_basis}",
                    f"  source_refs={','.join(incidence.source_refs) or 'none'}",
                    f"  lineage_source_ids={','.join(incidence.lineage_source_ids) or 'none'} lineage_tool_call_ids={','.join(incidence.lineage_tool_call_ids) or 'none'}",
                    f"  {projection.memory_content.get(incidence.memory_id, '')}",
                )
            )
            continue
        event_label = incidence.event_id or "none"
        if index:
            lines.extend(
                (
                    f"- object [{object_label}]",
                    f"  - temporal [{incidence.temporal_stratum}]",
                    f"    - event [{event_label}]",
                    f"      - cross-reference evidence [{incidence.memory_id}] route={incidence.route_basis}",
                )
            )
            continue
        lines.extend(
            (
                f"- object [{object_label}]",
                f"  - temporal [{incidence.temporal_stratum}]",
                f"    - event [{event_label}]",
                f"      - evidence [{incidence.memory_id}] route={incidence.route_basis} binding_basis={incidence.binding_basis}",
                f"        source_refs={','.join(incidence.source_refs) or 'none'}",
                f"        lineage_source_ids={','.join(incidence.lineage_source_ids) or 'none'} lineage_tool_call_ids={','.join(incidence.lineage_tool_call_ids) or 'none'}",
                f"        {projection.memory_content.get(incidence.memory_id, '')}",
            )
        )
    return tuple(lines)


def _membership_shell_lines(
    projection: Projection, incidence: SupportIncidence
) -> tuple[str, ...]:
    return (
        f"- membership [{incidence.memory_id}] requirement={incidence.requirement_id} object={_object_label(projection, incidence.object_scope_id)} event={incidence.event_id or 'none'} temporal={incidence.temporal_stratum} route={incidence.route_basis} binding_basis={incidence.binding_basis}",
    )


def _fallback_membership_shell_lines(
    projection: Projection, *, style: str
) -> tuple[str, ...]:
    if style == "full":
        grouped: dict[tuple[object, ...], list[str]] = {}
        for incidence in projection.incidences:
            key = (
                incidence.memory_id,
                incidence.object_scope_id,
                incidence.event_id,
                incidence.temporal_stratum,
                incidence.route_basis,
                incidence.binding_basis,
            )
            requirements = grouped.setdefault(key, [])
            if incidence.requirement_id not in requirements:
                requirements.append(incidence.requirement_id)
        lines: list[str] = []
        for (
            memory_id,
            object_scope_id,
            event_id,
            temporal_stratum,
            route_basis,
            binding_basis,
        ), requirement_ids in grouped.items():
            requirement_key = (
                "requirement" if len(requirement_ids) == 1 else "requirements"
            )
            lines.append(
                f"- membership [{memory_id}] {requirement_key}={','.join(requirement_ids)} object={_object_label(projection, object_scope_id)} event={event_id or 'none'} temporal={temporal_stratum} route={route_basis} binding_basis={binding_basis}"
            )
        return tuple(lines)
    if style == "compact":
        grouped_compact: dict[str, list[SupportIncidence]] = {}
        for incidence in projection.incidences:
            grouped_compact.setdefault(incidence.memory_id, []).append(incidence)

        def values(items: Sequence[object]) -> str:
            return ",".join(
                dict.fromkeys(("none" if item is None else str(item) for item in items))
            )

        return tuple(
            (
                f"- membership m[{memory_id}] r[{values((item.requirement_id for item in incidences))}] o[{values((_object_label(projection, item.object_scope_id) for item in incidences))}] e[{values((item.event_id for item in incidences))}] t[{values((item.temporal_stratum for item in incidences))}]"
                for memory_id, incidences in grouped_compact.items()
            )
        )
    if style == "minimal":
        grouped_minimal: dict[str, list[str]] = {}
        for incidence in projection.incidences:
            requirements = grouped_minimal.setdefault(incidence.memory_id, [])
            if incidence.requirement_id not in requirements:
                requirements.append(incidence.requirement_id)
        return tuple(
            (
                f"- membership m[{memory_id}] r[{','.join(requirement_ids)}]"
                for memory_id, requirement_ids in grouped_minimal.items()
            )
        )
    raise ValueError(f"unknown fallback membership shell style: {style}")


def _fallback_requirement_membership_shell_lines(
    projection: Projection, requirement_id: str, *, style: str
) -> tuple[str, ...]:
    incidences = projection.incidences_for_requirement(requirement_id)
    if not incidences:
        return ()
    if style == "full":
        return tuple(
            (
                line
                for incidence in incidences
                for line in _membership_shell_lines(projection, incidence)
            )
        )
    memory_ids = tuple(dict.fromkeys((incidence.memory_id for incidence in incidences)))
    if style == "compact":
        return (
            f"- membership map requirement={requirement_id} memories={','.join(memory_ids)}",
        )
    if style == "minimal":
        return (f"- membership map r[{requirement_id}] m[{','.join(memory_ids)}]",)
    raise ValueError(f"unknown fallback requirement shell style: {style}")


def _shared_candidate_lines(
    projection: Projection, memory_id: str, max_chars: int
) -> tuple[str, ...]:
    content, _ = _shorten_source_text(
        projection.memory_content.get(memory_id, ""),
        max_chars=max_chars,
        node_id=f"evidence:{memory_id}",
        source_id=memory_id,
        original_start_offset=0,
    )
    refs = ",".join(projection.memory_source_refs.get(memory_id, ())) or "none"
    evidence_id = f"evidence:{memory_id}"
    event_ids = tuple(
        (
            edge.parent_id.split(":", 1)[1]
            for edge in projection.edges
            if edge.kind == "event_evidence" and edge.child_id == evidence_id
        )
    )
    object_ids = tuple(
        (
            edge.parent_id.split(":", 1)[1]
            for edge in projection.edges
            if edge.kind == "object_evidence" and edge.child_id == evidence_id
        )
    )
    objects = tuple(
        (
            node.label
            for node in projection.nodes
            if node.id in {f"object:{item}" for item in object_ids}
        )
    )
    return (
        f"- candidate [{memory_id}] binding_basis=unassigned_candidate",
        f"  object={','.join(objects) or 'unscoped'} event={','.join(event_ids) or 'none'} source_refs={refs}",
        f"  {content}",
    )


def _unassigned_lines(
    projection: Projection,
    memory_id: str,
    max_chars: int,
    *,
    provider_name: str = "event_stage",
) -> tuple[str, ...]:
    refs = ",".join(projection.memory_source_refs.get(memory_id, ())) or "none"
    object_scope_ids = _current_object_scope_ids(projection, memory_id) or (None,)
    if provider_name == "flat":
        lines = [
            f"- current-event evidence [{memory_id}] object={_object_label(projection, object_scope_ids[0])} route=current_event_visibility",
            f"  source_refs={refs}",
            f"  {projection.memory_content.get(memory_id, '')}",
        ]
        lines.extend(
            (
                f"- current-event cross-reference [{memory_id}] object={_object_label(projection, object_scope_id)} route=current_event_visibility"
                for object_scope_id in object_scope_ids[1:]
            )
        )
        return tuple(lines)
    lines: list[str] = []
    for index, object_scope_id in enumerate(object_scope_ids):
        lines.extend(
            (
                f"- object [{_object_label(projection, object_scope_id)}]",
                "  - temporal [current_event]",
                f"    - event [{projection.current_event_id or 'none'}]",
                f"      - evidence [{memory_id}] route=current_event_visibility"
                if index == 0
                else f"      - cross-reference evidence [{memory_id}] route=current_event_visibility",
            )
        )
        if index == 0:
            lines.extend(
                (
                    f"        source_refs={refs}",
                    f"        {projection.memory_content.get(memory_id, '')}",
                )
            )
    return tuple(lines)


def _current_event_cross_reference_line(
    projection: Projection,
    memory_id: str,
    *,
    owner_requirement_id: str,
    provider_name: str = "event_stage",
) -> tuple[str, ...]:
    object_scope_ids = _current_object_scope_ids(projection, memory_id) or (None,)
    if provider_name == "flat":
        return tuple(
            (
                f"- current-event visibility for memory [{memory_id}]; object={_object_label(projection, object_scope_id)}; full atom rendered under requirement {owner_requirement_id}"
                for object_scope_id in object_scope_ids
            )
        )
    lines: list[str] = []
    for object_scope_id in object_scope_ids:
        lines.extend(
            (
                f"- object [{_object_label(projection, object_scope_id)}]",
                "  - temporal [current_event]",
                f"    - event [{projection.current_event_id or 'none'}]",
                f"      - current-event visibility for memory [{memory_id}]; full atom rendered under requirement {owner_requirement_id}",
            )
        )
    return tuple(lines)


def _current_object_labels(projection: Projection, memory_id: str) -> tuple[str, ...]:
    return tuple(
        (
            _object_label(projection, object_scope_id)
            for object_scope_id in _current_object_scope_ids(projection, memory_id)
        )
    )


def _current_object_scope_ids(
    projection: Projection, memory_id: str
) -> tuple[str, ...]:
    evidence_id = f"evidence:{memory_id}"
    object_node_ids = {
        edge.parent_id
        for edge in projection.edges
        if edge.kind == "object_evidence"
        and edge.child_id == evidence_id
        and (edge.provenance == "current_event_visibility")
    }
    return tuple(
        sorted(
            (
                node.id.split(":", 1)[1]
                for node in projection.nodes
                if node.id in object_node_ids and node.kind == "object"
            )
        )
    )


def _object_label(projection: Projection, object_scope_id: str | None) -> str:
    if object_scope_id is None:
        return "unscoped"
    node_id = f"object:{object_scope_id}"
    return next(
        (node.label for node in projection.nodes if node.id == node_id), object_scope_id
    )


def _cross_reference_line(
    memory_id: str, *, owner_requirement_id: str, overlap_requirement_ids: Sequence[str]
) -> str:
    overlap = ",".join(overlap_requirement_ids) or owner_requirement_id
    return f"- cross-reference memory [{memory_id}] rendered under requirement {owner_requirement_id}; overlap_requirements={overlap}"


def _cross_reference_lines(
    projection: Projection,
    incidences: Sequence[SupportIncidence],
    *,
    memory_id: str,
    owner_requirement_id: str,
    overlap_requirement_ids: Sequence[str],
) -> tuple[str, ...]:
    overlap = ",".join(overlap_requirement_ids) or owner_requirement_id
    return tuple(
        (
            f"- cross-reference memory [{memory_id}] object={_object_label(projection, incidence.object_scope_id)} event={incidence.event_id or 'none'} rendered under requirement {owner_requirement_id}; overlap_requirements={overlap}"
            for incidence in incidences
        )
    )


def _overlap_requirement_ids(projection: Projection, memory_id: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                incidence.requirement_id
                for incidence in projection.incidences
                if incidence.memory_id == memory_id
            )
        )
    )


def _requirement_queues(
    projection: Projection, provider: HistoryProvider
) -> dict[str, tuple[str, ...]]:
    queues: dict[str, tuple[str, ...]] = {}
    for requirement in projection.requirements:
        memory_ids = tuple(
            dict.fromkeys(
                (
                    incidence.memory_id
                    for incidence in projection.incidences_for_requirement(
                        requirement.id
                    )
                )
            )
        )
        queues[requirement.id] = tuple(
            (item.memory_id for item in provider.order(projection, memory_ids))
        )
    return queues


def _round_robin_pairs(
    queues: Mapping[str, Sequence[str]],
) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    queue_items = tuple(queues.items())
    for index in range(max((len(items) for _, items in queue_items), default=0)):
        for requirement_id, items in queue_items:
            if index >= len(items):
                continue
            pair = (requirement_id, items[index])
            if pair not in seen:
                seen.add(pair)
                result.append(pair)
    return tuple(result)


def _incidence_for_memory(
    incidences: Sequence[SupportIncidence], memory_id: str
) -> SupportIncidence | None:
    return next(
        (incidence for incidence in incidences if incidence.memory_id == memory_id),
        None,
    )


def _incidences_for_memory(
    incidences: Sequence[SupportIncidence], memory_id: str
) -> tuple[SupportIncidence, ...]:
    seen: set[tuple[str | None, str | None, str, str, tuple[str, ...]]] = set()
    result: list[SupportIncidence] = []
    for incidence in incidences:
        if incidence.memory_id != memory_id:
            continue
        key = (
            incidence.object_scope_id,
            incidence.event_id,
            incidence.temporal_stratum,
            incidence.route_basis,
            incidence.source_refs,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(incidence)
    return tuple(result)


def _build_memberships(
    projection: Projection,
    *,
    rendered_modes: Mapping[
        tuple[str | None, str | None, str | None, str, str, str], str
    ],
) -> tuple[EvidenceMembership, ...]:
    requirement_ids_by_memory: dict[str, list[str]] = {}
    branch_keys: list[tuple[str | None, str | None, str | None, str, str, str]] = []
    for incidence in projection.incidences:
        requirement_ids_by_memory.setdefault(incidence.memory_id, [])
        if (
            incidence.requirement_id
            not in requirement_ids_by_memory[incidence.memory_id]
        ):
            requirement_ids_by_memory[incidence.memory_id].append(
                incidence.requirement_id
            )
        branch_keys.append(_incidence_branch_key(incidence))
    for memory_id in projection.current_event_memory_ids:
        branch_keys.extend(_current_branch_keys(projection, memory_id))
    result: list[EvidenceMembership] = []
    for branch_key in dict.fromkeys(branch_keys):
        (
            requirement_id,
            object_scope_id,
            event_id,
            temporal_stratum,
            memory_id,
            route,
        ) = branch_key
        all_requirement_ids = tuple(requirement_ids_by_memory.get(memory_id, ()))
        overlap = tuple(
            (item for item in all_requirement_ids if item != requirement_id)
        )
        mode = rendered_modes.get(branch_key)
        incidence = next(
            (
                incidence
                for incidence in projection.incidences
                if _incidence_branch_key(incidence) == branch_key
            ),
            None,
        )
        result.append(
            EvidenceMembership(
                requirement_id=requirement_id,
                object_scope_id=object_scope_id,
                event_id=event_id,
                temporal_stratum=temporal_stratum,
                memory_id=memory_id,
                route=route,
                rendered_as_evidence=mode == "evidence",
                cross_reference=mode == "cross_reference",
                rendered_as_membership_shell=mode == "membership_shell",
                omitted=mode is None,
                omission_reason="omitted_by_budget" if mode is None else None,
                overlap_requirement_ids=overlap,
                binding_basis=incidence.binding_basis
                if incidence is not None
                else None,
                lineage_source_ids=incidence.lineage_source_ids
                if incidence is not None
                else (),
                lineage_tool_call_ids=incidence.lineage_tool_call_ids
                if incidence is not None
                else (),
            )
        )
    return tuple(result)


def _incidence_branch_key(
    incidence: SupportIncidence,
) -> tuple[str | None, str | None, str | None, str, str, str]:
    return (
        incidence.requirement_id,
        incidence.object_scope_id,
        incidence.event_id,
        incidence.temporal_stratum,
        incidence.memory_id,
        incidence.route_basis,
    )


def _current_branch_keys(
    projection: Projection, memory_id: str
) -> tuple[tuple[str | None, str | None, str | None, str, str, str], ...]:
    scopes: tuple[str | None, ...] = _current_object_scope_ids(
        projection, memory_id
    ) or (None,)
    return tuple(
        (
            (
                None,
                object_scope_id,
                projection.current_event_id,
                TEMPORAL_CURRENT,
                memory_id,
                "current_event_visibility",
            )
            for object_scope_id in scopes
        )
    )


def _first_incidence_by_memory(
    incidences: Sequence[SupportIncidence],
) -> dict[str, SupportIncidence]:
    return {
        memory_id: incidence
        for memory_id, incidence in reversed(
            [(incidence.memory_id, incidence) for incidence in incidences]
        )
    }


def _shorten_source_text(
    text: str,
    *,
    max_chars: int,
    node_id: str,
    source_id: str,
    original_start_offset: int,
) -> tuple[str, TextOmission | None]:
    original = str(text)
    if len(original) <= max_chars:
        return (original, None)
    rendered_length = max(0, max_chars - 3)
    rendered = original[:rendered_length] + "..."
    return (
        rendered,
        TextOmission(
            node_id=node_id,
            source_id=source_id,
            original_start_offset=original_start_offset,
            original_end_offset=original_start_offset + len(original),
            rendered_start_offset=original_start_offset,
            rendered_end_offset=original_start_offset + rendered_length,
            content_sha256=sha256(original.encode("utf-8")).hexdigest(),
            reason="max_chars",
        ),
    )


def _all_source_text_omissions(
    projection: Projection, *, reason: str
) -> tuple[TextOmission, ...]:
    records = [
        TextOmission(
            node_id=f"task:{projection.task_source_id}",
            source_id=projection.task_source_id,
            original_start_offset=0,
            original_end_offset=len(projection.task_text),
            rendered_start_offset=0,
            rendered_end_offset=0,
            content_sha256=sha256(projection.task_text.encode("utf-8")).hexdigest(),
            reason=reason,
        )
    ]
    records.extend(
        (
            TextOmission(
                node_id=f"requirement:{requirement.id}",
                source_id=requirement.source_id,
                original_start_offset=requirement.start_offset,
                original_end_offset=requirement.end_offset,
                rendered_start_offset=requirement.start_offset,
                rendered_end_offset=requirement.start_offset,
                content_sha256=sha256(
                    requirement.exact_text.encode("utf-8")
                ).hexdigest(),
                reason=reason,
            )
            for requirement in projection.requirements
        )
    )
    return tuple(records)


def _candidate_memory_ids(projection: Projection) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *(incidence.memory_id for incidence in projection.incidences),
                *projection.current_event_memory_ids,
                *projection.unassigned_current_memory_ids,
                *projection.shared_candidate_memory_ids,
            )
        )
    )


def _render_budget_fallback(
    projection: Projection,
    *,
    candidate_ids: Sequence[str],
    current_event_ids: Sequence[str],
    provider_name: str,
    max_tokens: int,
    token_counter: TokenCounter,
    memory_line_max_chars: int,
) -> tuple[
    str,
    list[str],
    list[str],
    dict[tuple[str | None, str | None, str | None, str, str, str], str],
]:
    candidate_ids = tuple(dict.fromkeys(candidate_ids))
    current_event_ids = tuple(dict.fromkeys(current_event_ids))
    current_set = set(current_event_ids)
    priority = (
        *current_event_ids,
        *(memory_id for memory_id in candidate_ids if memory_id not in current_set),
    )
    provider = (
        FlatHistoryProvider()
        if provider_name == "flat"
        else EventStageHistoryProvider()
    )
    owner_by_memory: dict[str, str] = {}
    for requirement in projection.requirements:
        for memory_id in _requirement_queues(projection, provider).get(
            requirement.id, ()
        ):
            if _incidences_for_memory(
                projection.incidences_for_requirement(requirement.id), memory_id
            ):
                owner_by_memory.setdefault(memory_id, requirement.id)
    membership_shell_style = "full"

    def build(
        selected: Sequence[str],
        *,
        include_ledger: bool,
        include_current_omissions: bool,
    ) -> tuple[
        str,
        dict[tuple[str | None, str | None, str | None, str, str, str], str],
        list[str],
        set[str],
    ]:
        selected_set = set(selected)
        omitted = [
            memory_id for memory_id in candidate_ids if memory_id not in selected_set
        ]
        lines = [
            "[ACG task-state projection]",
            f"[Task source] {projection.task_source_id}",
            "[Requirement IDs] "
            + (
                ",".join((requirement.id for requirement in projection.requirements))
                or "none"
            ),
            f"[Hierarchy] {provider_name}",
            f"[Current event] {projection.current_event_id or 'none'}",
        ]
        modes: dict[tuple[str | None, str | None, str | None, str, str, str], str] = {}
        visible: list[str] = []
        body_rendered_ids: set[str] = set()
        membership_shell_lines = _fallback_membership_shell_lines(
            projection, style=membership_shell_style
        )
        if membership_shell_style != "full" and membership_shell_lines:
            lines.extend(("[Requirement membership index]", *membership_shell_lines))
        if projection.current_event_id is not None or current_event_ids:
            lines.append("[Current event]")
            for memory_id in current_event_ids:
                if memory_id in selected_set:
                    lines.extend(
                        _unassigned_lines(
                            projection,
                            memory_id,
                            memory_line_max_chars,
                            provider_name=provider_name,
                        )
                    )
                    visible.append(memory_id)
                    body_rendered_ids.add(memory_id)
                    for index, key in enumerate(
                        _current_branch_keys(projection, memory_id)
                    ):
                        modes[key] = "evidence" if index == 0 else "cross_reference"
                elif include_current_omissions:
                    for object_scope_id in _current_object_scope_ids(
                        projection, memory_id
                    ) or (None,):
                        lines.append(
                            f"- object [{_object_label(projection, object_scope_id)}] temporal [current_event] event [{projection.current_event_id or 'none'}] evidence [{memory_id}] omitted_by_budget"
                        )
        for requirement in projection.requirements:
            lines.extend(("", f"[Requirement {requirement.id} support]"))
            incidences = projection.incidences_for_requirement(requirement.id)
            if not incidences:
                lines.append(
                    "- no source-grounded support record is visible in this projection"
                )
            else:
                lines.extend(
                    _fallback_requirement_membership_shell_lines(
                        projection, requirement.id, style=membership_shell_style
                    )
                )
            for incidence in incidences:
                key = _incidence_branch_key(incidence)
                modes.setdefault(key, "membership_shell")
            memory_ids = tuple(dict.fromkeys((item.memory_id for item in incidences)))
            for memory_id in memory_ids:
                if memory_id not in selected_set or memory_id in current_set:
                    continue
                branch_incidences = _incidences_for_memory(incidences, memory_id)
                if owner_by_memory.get(memory_id) != requirement.id:
                    continue
                lines.extend(
                    _evidence_lines(
                        projection,
                        branch_incidences,
                        memory_line_max_chars,
                        provider_name=provider_name,
                    )
                )
                body_rendered_ids.add(memory_id)
                for index, incidence in enumerate(branch_incidences):
                    modes[_incidence_branch_key(incidence)] = (
                        "evidence" if index == 0 else "cross_reference"
                    )
        if projection.unassigned_candidate_memory_ids:
            lines.append("[Unassigned candidate evidence]")
            for memory_id in projection.unassigned_candidate_memory_ids:
                if memory_id in selected_set:
                    lines.extend(
                        _shared_candidate_lines(
                            projection, memory_id, memory_line_max_chars
                        )
                    )
        if include_ledger and omitted:
            lines.append("[Evidence omission ledger]")
            lines.extend(
                (
                    f"- evidence [{memory_id}] omitted_by_budget; source refs remain in the graph"
                    for memory_id in omitted
                )
            )
        return ("\n".join(lines), modes, visible, body_rendered_ids)

    shell_text = ""
    shell_modes: dict[
        tuple[str | None, str | None, str | None, str, str, str], str
    ] = {}
    shell_visible: list[str] = []
    for shell_style in ("full", "compact", "minimal"):
        membership_shell_style = shell_style
        shell_text, shell_modes, shell_visible, _ = build(
            (), include_ledger=False, include_current_omissions=False
        )
        if token_counter(shell_text) <= max_tokens:
            break
    if token_counter(shell_text) > max_tokens:
        return _render_overflow_index(
            projection,
            candidate_ids=candidate_ids,
            current_event_ids=current_event_ids,
            provider_name=provider_name,
            max_tokens=max_tokens,
            token_counter=token_counter,
        )

    def best_text(selected: Sequence[str]) -> tuple[str, dict, list[str], set[str]]:
        for include_current_omissions, include_ledger in (
            (True, True),
            (False, True),
            (False, False),
        ):
            candidate_text, candidate_modes, candidate_visible, candidate_body_ids = (
                build(
                    selected,
                    include_ledger=include_ledger,
                    include_current_omissions=include_current_omissions,
                )
            )
            if token_counter(candidate_text) <= max_tokens:
                return (
                    candidate_text,
                    candidate_modes,
                    candidate_visible,
                    candidate_body_ids,
                )
        return (shell_text, shell_modes, shell_visible, set())

    text, modes, visible, _ = best_text(())
    selected: list[str] = []
    for memory_id in priority:
        candidate = [*selected, memory_id]
        candidate_text, candidate_modes, candidate_visible, candidate_body_ids = (
            best_text(candidate)
        )
        if memory_id in candidate_body_ids:
            selected = candidate
            text, modes, visible = (candidate_text, candidate_modes, candidate_visible)
    return (text, selected, visible, modes)


def _render_overflow_index(
    projection: Projection,
    *,
    candidate_ids: Sequence[str],
    current_event_ids: Sequence[str],
    provider_name: str,
    max_tokens: int,
    token_counter: TokenCounter,
) -> tuple[str, list[str], list[str], dict]:
    symbols: list[str] = []
    symbol_ids: dict[str, int] = {}

    def ref(value: str | None) -> int | None:
        if value is None:
            return None
        if value not in symbol_ids:
            symbol_ids[value] = len(symbols)
            symbols.append(value)
        return symbol_ids[value]

    requirements = [
        [ref(r.id), ref(r.source_id), r.start_offset, r.end_offset]
        for r in projection.requirements
    ]
    strata = {
        TEMPORAL_CURRENT: "c",
        TEMPORAL_LATEST: "l",
        TEMPORAL_EARLIER: "e",
        TEMPORAL_UNSCOPED: "u",
    }
    rows = [
        [
            ref(i.requirement_id),
            ref(i.object_scope_id),
            ref(i.memory_id),
            ref(i.event_id),
            strata.get(i.temporal_stratum, i.temporal_stratum),
        ]
        for i in projection.incidences
    ]
    payload = {
        "task": ref(projection.task_source_id),
        "hierarchy": provider_name,
        "requirements": requirements,
        "support": rows,
        "current_event": ref(projection.current_event_id),
        "current_memories": [ref(m) for m in current_event_ids],
        "memory_sources": [
            [
                ref(m),
                [
                    ref(s)
                    for s in dict.fromkeys(
                        (
                            *projection.memory_source_refs.get(m, ()),
                            *(
                                s
                                for i in projection.incidences
                                if i.memory_id == m
                                for s in i.source_refs
                            ),
                        )
                    )
                ],
            ]
            for m in candidate_ids
        ],
        "symbols": symbols,
    }
    header = "[ACG overflow index]\nIDs index symbols. requirements=[id,source,start,end]; support=[req,object,memory,event,time]; time c=current,l=latest,e=earlier,u=unscoped.\nLinks do not decide completion. Unlisted bodies remain in graph state.\n"
    text = header + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    packed_memory_indices: dict[str, int] | None = None
    if token_counter(text) > max_tokens:
        symbols.clear()
        symbol_ids.clear()
        payload = {
            "task": ref(projection.task_source_id),
            "requirements": [
                [ref(r.id), ref(r.source_id), r.start_offset, r.end_offset]
                for r in projection.requirements
            ],
            "support": [
                [
                    ref(r.id),
                    [
                        ref(m)
                        for m in dict.fromkeys(
                            (
                                i.memory_id
                                for i in projection.incidences
                                if i.requirement_id == r.id
                            )
                        )
                    ],
                ]
                for r in projection.requirements
            ],
            "memory_sources": [
                [
                    ref(m),
                    [
                        ref(s)
                        for s in dict.fromkeys(
                            (
                                *projection.memory_source_refs.get(m, ()),
                                *(
                                    s
                                    for i in projection.incidences
                                    if i.memory_id == m
                                    for s in i.source_refs
                                ),
                            )
                        )
                    ],
                ]
                for m in candidate_ids
            ],
            "symbols": symbols,
        }
        header = "[ACG overflow index]\nIDs index symbols; requirements=[id,source,start,end]; support=[req,memories].\nLinks do not decide completion. Unlisted bodies remain in graph state.\n"
        text = header + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if token_counter(text) > max_tokens:
            text, packed_memory_indices = _render_readable_packed_index(
                projection,
                candidate_ids=candidate_ids,
                current_event_ids=current_event_ids,
                provider_name=provider_name,
            )
            if token_counter(text) > max_tokens:
                raise RenderBudgetError(
                    "mandatory requirement membership quick index exceeds render budget"
                )
    selected: list[str] = []
    modes = {
        _incidence_branch_key(i): "membership_shell" for i in projection.incidences
    }
    for memory_id in dict.fromkeys((*current_event_ids, *candidate_ids)):
        content = projection.memory_content.get(memory_id)
        if not content:
            continue
        if packed_memory_indices is None:
            line = "\nevidence " + json.dumps(
                [ref(memory_id), content], ensure_ascii=False
            )
        else:
            line = "\nB" + json.dumps(
                [packed_memory_indices[memory_id], content], ensure_ascii=False
            )
        if token_counter(text + line) > max_tokens:
            continue
        text += line
        selected.append(memory_id)
        branches = [
            _incidence_branch_key(i)
            for i in projection.incidences
            if i.memory_id == memory_id
        ]
        branches.extend(
            _current_branch_keys(projection, memory_id)
            if memory_id in current_event_ids
            else ()
        )
        for index, branch in enumerate(branches):
            modes[branch] = "evidence" if index == 0 else "cross_reference"
    return (text, selected, [m for m in current_event_ids if m in selected], modes)


def _render_readable_packed_index(
    projection: Projection,
    *,
    candidate_ids: Sequence[str],
    current_event_ids: Sequence[str],
    provider_name: str,
) -> tuple[str, dict[str, int]]:

    def ordered(values: Sequence[str | None]) -> list[str]:
        return list(dict.fromkeys((value for value in values if value is not None)))

    def registry(
        values: Sequence[str], prefixes: Sequence[str]
    ) -> tuple[list[object], dict[str, int]]:
        values = ordered(values)
        common = os.path.commonprefix(values) if values else ""
        cut = max((common.rfind(marker) for marker in ("-", "_", ":", "/", ".")))
        if cut >= 0:
            common = common[: cut + 1]
        candidates = [
            item
            for item in (*prefixes, common)
            if item and values and all((value.startswith(item) for value in values))
        ]
        prefix = max(candidates, key=len, default="")
        return (
            [prefix, [value[len(prefix) :] for value in values]],
            {value: index for index, value in enumerate(values)},
        )

    requirement_ids = ordered([item.id for item in projection.requirements])
    memory_ids = ordered(
        [
            *candidate_ids,
            *current_event_ids,
            *projection.current_event_memory_ids,
            *projection.unassigned_current_memory_ids,
            *projection.unassigned_candidate_memory_ids,
            *(item.memory_id for item in projection.incidences),
        ]
    )
    source_ids = ordered(
        [
            projection.task_source_id,
            *(item.source_id for item in projection.requirements),
            *(ref for item in projection.incidences for ref in item.source_refs),
            *(
                ref
                for memory_id in memory_ids
                for ref in projection.memory_source_refs.get(memory_id, ())
            ),
        ]
    )
    current_memory_ids = tuple(
        dict.fromkeys(
            (
                *current_event_ids,
                *projection.current_event_memory_ids,
                *projection.unassigned_current_memory_ids,
            )
        )
    )
    current_object_scopes = {
        memory_id: _current_object_scope_ids(projection, memory_id)
        for memory_id in current_memory_ids
    }
    object_ids = ordered(
        [
            *(item.object_scope_id for item in projection.incidences),
            *(
                object_scope_id
                for object_scope_ids in current_object_scopes.values()
                for object_scope_id in object_scope_ids
            ),
        ]
    )
    event_ids = ordered(
        [
            projection.current_event_id,
            *(item.event_id for item in projection.incidences),
        ]
    )
    req_registry, req_index = registry(requirement_ids, ("R", "requirement-"))
    source_registry, source_index = registry(source_ids, ("src-", "source-"))
    memory_registry, memory_index = registry(memory_ids, ("mem-", "memory-"))
    object_registry, object_index = registry(object_ids, ("object-", "obj-"))
    event_registry, event_index = registry(event_ids, ("event-",))
    strata = {
        TEMPORAL_CURRENT: 0,
        TEMPORAL_LATEST: 1,
        TEMPORAL_EARLIER: 2,
        TEMPORAL_UNSCOPED: 3,
    }
    incidences = [
        [
            req_index.get(item.requirement_id, -1),
            object_index.get(item.object_scope_id, -1),
            memory_index[item.memory_id],
            event_index.get(item.event_id, -1),
            strata.get(item.temporal_stratum, item.temporal_stratum),
        ]
        for item in projection.incidences
    ]
    requirements = [
        [
            req_index[item.id],
            source_index[item.source_id],
            item.start_offset,
            item.end_offset,
        ]
        for item in projection.requirements
    ]
    memory_sources_rows = [
        [
            memory_index[memory_id],
            [
                source_index[source_id]
                for source_id in ordered(
                    (
                        *projection.memory_source_refs.get(memory_id, ()),
                        *(
                            source
                            for item in projection.incidences
                            if item.memory_id == memory_id
                            for source in item.source_refs
                        ),
                    )
                )
            ],
        ]
        for memory_id in memory_ids
    ]
    source_offsets = {
        refs[0] - memory_id for memory_id, refs in memory_sources_rows if len(refs) == 1
    }
    if len(source_offsets) == 1 and all(
        (len(refs) == 1 for _, refs in memory_sources_rows)
    ):
        memory_sources: object = ["+", source_offsets.pop()]
    else:
        memory_sources = memory_sources_rows
    current_objects = [
        [
            memory_index[memory_id],
            [object_index[item] for item in current_object_scopes[memory_id]],
        ]
        for memory_id in current_event_ids
        if memory_id in memory_index
    ]
    payload = {
        "V": 1,
        "H": provider_name,
        "T": source_index[projection.task_source_id],
        "Q": requirements,
        "I": incidences,
        "C": [
            event_index[projection.current_event_id]
            if projection.current_event_id is not None
            else -1,
            [memory_index[item] for item in current_event_ids if item in memory_index],
            [
                memory_index[item]
                for item in projection.unassigned_current_memory_ids
                if item in memory_index
            ],
        ],
        "O": current_objects,
        "X": memory_sources,
        "D": [
            ["r", *req_registry],
            ["s", *source_registry],
            ["m", *memory_registry],
            ["o", *object_registry],
            ["e", *event_registry],
        ],
    }
    header = "[ACG packed index]\nD=[tag,prefix,suffixes], ID=prefix+suffixes[index]; Q=[r,s,start,end]; I=[r,o,m,e,time] where time 0=current,1=latest,2=earlier,3=unscoped and missing r/o/e=-1; C=[event,current_memories,unassigned_memories] with missing event=-1; O=[memory,objects]; X=[memory,sources] or [+,offset] meaning memory i has source i+offset; B=[memory,complete_body].\nThis is a lossless structural membership/source index. Links do not decide completion. Bodies, route, binding, and lineage remain in graph state.\n"
    return (
        header + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        memory_index,
    )


def _fit_compact_shell(
    projection: Projection,
    *,
    candidate_ids: Sequence[str],
    current_event_ids: Sequence[str],
    provider_name: str,
    max_tokens: int,
    token_counter: TokenCounter,
) -> str:
    lines = [
        "[ACG compact index]",
        f"task:{projection.task_source_id}",
        "requirements:"
        + (",".join((item.id for item in projection.requirements)) or "none"),
        f"hierarchy:{provider_name}",
        f"current_event:{projection.current_event_id or 'none'}",
    ]
    for memory_id in current_event_ids:
        object_scope_ids = _current_object_scope_ids(projection, memory_id) or (None,)
        for object_scope_id in object_scope_ids:
            lines.append(
                f"object:{_object_label(projection, object_scope_id)} temporal:current_event event:{projection.current_event_id or 'none'} evidence:{memory_id}:omitted_by_budget"
            )
    lines.append("omitted_by_budget:")
    lines.extend((f"evidence:{memory_id}" for memory_id in candidate_ids))
    return _fit_lines(lines, max_tokens=max_tokens, token_counter=token_counter)


def _fit_with_omission_priority(
    omitted_memory_ids: Sequence[str], *, max_tokens: int, token_counter: TokenCounter
) -> str:
    ids = tuple(dict.fromkeys(omitted_memory_ids))
    if not ids:
        return _fit_lines(
            ("[ACG task-state projection]",),
            max_tokens=max_tokens,
            token_counter=token_counter,
        )
    lines = ["[ACG task-state projection]", "[Omitted evidence]"]
    for memory_id in ids:
        detail = f"- {memory_id}: omitted_by_budget; source refs remain in the graph"
        candidate = "\n".join((*lines, detail))
        if token_counter(candidate) <= max_tokens:
            lines.append(detail)
            continue
        compact = f"- {memory_id}: omitted_by_budget; refs=graph"
        candidate = "\n".join((*lines, compact))
        if token_counter(candidate) <= max_tokens:
            lines.append(compact)
            continue
        break
    rendered_ids = len(lines) - 2
    remaining = len(ids) - rendered_ids
    if remaining:
        summary = f"- +{remaining} omitted IDs; reason=omitted_by_budget; refs=graph"
        candidate = "\n".join((*lines, summary))
        if token_counter(candidate) <= max_tokens:
            lines.append(summary)
    text = "\n".join(lines)
    if token_counter(text) <= max_tokens:
        return text
    return _fit_lines(
        ("[ACG acg]", "[Omitted evidence]"),
        max_tokens=max_tokens,
        token_counter=token_counter,
    )


def _fit_lines(
    lines: Sequence[str], *, max_tokens: int, token_counter: TokenCounter
) -> str:
    selected: list[str] = []
    for line in lines:
        candidate = "\n".join((*selected, line))
        if token_counter(candidate) > max_tokens:
            break
        selected.append(line)
    return "\n".join(selected)


__all__ = [
    "EvidenceMembership",
    "RenderedContext",
    "TextOmission",
    "fit_projection",
    "render_projection",
]
