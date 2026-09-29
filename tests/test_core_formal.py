import json

import pytest
from acg.core import (
    ACGConfig,
    AdaptiveConsistencyGraph,
    LLMConceptIndexer,
    RenderBudgetError,
    SourceKind,
)
from acg.core.concepts import (
    ConceptContext,
    ConceptRecord,
    ConceptUnit,
    EventOrganization,
    concept_output_schema,
    parse_concept_output,
)
from acg.core.projection import Projection, SupportIncidence
from acg.core.renderer import render_projection
from acg.core.schema import SourceRecord
from acg.core.source_context import RequirementAnchor


class Embeddings:
    def embed(self, texts):
        return [self.vector(text) for text in texts]

    @staticmethod
    def vector(text):
        value = text.lower()
        return (
            1.0 + value.count("blue"),
            1.0 + value.count("shirt"),
            1.0 + value.count("price"),
        )


class Client:
    max_tokens = 2048
    calls = 0
    failures = 0
    conversation_recorder = None

    def complete_json(self, **kwargs):
        self.calls += 1
        return {"records": []}


def populated_graph():
    graph = AdaptiveConsistencyGraph(
        embedding_provider=Embeddings(), config=ACGConfig(projection_tokens=5000)
    )
    graph.start("Find a blue shirt and report the price.")
    action = graph.record(
        SourceKind.ACTION,
        {
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "search",
                        "arguments": {"query": "blue shirt"},
                    },
                }
            ]
        },
    )
    observation = graph.record(
        SourceKind.OBSERVATION,
        {"tool_call_id": "call-1", "items": [{"name": "Blue Shirt", "price": "$20"}]},
    )
    graph.update([action.id, observation.id])
    return graph, action, observation


def test_formal_view_exposes_projection_and_rendered_context():
    graph, action, observation = populated_graph()
    view = graph.decision_view(
        recent_source_ids=[action.id, observation.id],
        latest_action_source_id=action.id,
        latest_observation_source_id=observation.id,
        available_memory_tokens=5000,
    )
    assert view.projection is not None
    assert view.rendered is not None
    assert "Blue Shirt" in view.rendered.text
    assert view.projection.hierarchy_provider == "event_stage"


def test_failure_retains_projection_and_snapshot_is_jsonable():
    graph, action, observation = populated_graph()
    with pytest.raises(RenderBudgetError):
        graph.decision_view(
            recent_source_ids=[action.id, observation.id],
            latest_action_source_id=action.id,
            latest_observation_source_id=observation.id,
            available_memory_tokens=1,
        )
    assert graph.failed_projection is not None
    snapshot = graph.snapshot()
    assert snapshot["schema_version"] == "acg.core.v1"
    json.dumps(snapshot)


def test_indexer_accepts_generic_structured_client():
    indexer = LLMConceptIndexer(Client())
    assert indexer.client.calls == 0


def dense_projection():
    count = 13
    requirements = tuple(
        RequirementAnchor(
            f"requirement-{index:02d}",
            "source-task-very-long-common-prefix",
            index * 10,
            index * 10 + 9,
            f"requirement {index}",
        )
        for index in range(1, 4)
    )
    memory_content = {
        f"memory-very-long-common-prefix-{index:03d}": "x"
        for index in range(1, count + 1)
    }
    incidences = tuple(
        SupportIncidence(
            requirement_id=f"requirement-{(index % 3) + 1:02d}",
            object_scope_id=f"object-very-long-common-prefix-{index % 2}",
            memory_id=f"memory-very-long-common-prefix-{index:03d}",
            event_id=f"event-very-long-common-prefix-{index % 3}",
            temporal_stratum="earlier",
            route_basis="retrieved",
            source_refs=(f"source-very-long-common-prefix-{index:03d}",),
        )
        for index in range(1, count + 1)
    )
    return Projection(
        task_source_id="source-task-very-long-common-prefix",
        task_text="task",
        requirements=requirements,
        nodes=(),
        edges=(),
        incidences=incidences,
        current_event_id="event-very-long-common-prefix-2",
        current_event_memory_ids=(),
        unassigned_current_memory_ids=(),
        memory_content=memory_content,
        memory_source_refs={
            memory_id: (f"source-very-long-common-prefix-{index:03d}",)
            for index, memory_id in enumerate(memory_content, start=1)
        },
        event_sequences={
            f"event-very-long-common-prefix-{index}": index for index in range(3)
        },
        omission_reasons={},
    )


def test_renderer_preserves_dense_memberships_across_budget_formats():
    projection = dense_projection()
    cases = (
        (5000, "renderer.v1", "[ACG task-state projection]"),
        (1600, "renderer.overflow_index.v1", "[ACG overflow index]"),
        (1500, "renderer.packed_index.v1", "[ACG packed index]"),
    )
    for budget, version, header in cases:
        rendered = render_projection(projection, max_tokens=budget, token_counter=len)
        assert rendered.renderer_version == version
        assert rendered.text.startswith(header)
        assert rendered.token_count <= budget
        assert len(rendered.memberships) == len(projection.incidences)
        assert {item.memory_id for item in rendered.memberships} == set(
            projection.memory_content
        )


def test_structured_concept_response_retains_assignments_and_rejects_unknown_fields():
    context = ConceptContext(
        task=SourceRecord("task-1", SourceKind.TASK, "Track product A.", 1),
        records=(
            ConceptRecord(
                "record-1", "action-1", "action", (ConceptUnit("unit-1", "product A"),)
            ),
        ),
        candidate_ids=(),
        organization=EventOrganization(),
    )
    payload = {
        "records": [
            {
                "record_id": "record-1",
                "existing_concept_ids": [],
                "new_anchors": [["unit-1"]],
            }
        ],
        "action_origins": [],
        "user_source_decisions": [],
    }
    proposal = parse_concept_output(payload, context=context)
    assert not proposal.parse_errors
    assert proposal.assignments[0].record_id == "record-1"
    assert proposal.assignments[0].new_anchor_texts == (("product A",),)
    schema = concept_output_schema(context)
    assert set(schema["required"]) == {
        "records",
        "action_origins",
        "user_source_decisions",
    }

    class StructuredClient:
        def complete_json(self, **request):
            assert set(request["schema"]["required"]) == {
                "records",
                "action_origins",
                "user_source_decisions",
            }
            return payload

    client_proposal = LLMConceptIndexer(StructuredClient()).propose(context)
    assert client_proposal.assignments == proposal.assignments
    malformed = parse_concept_output({**payload, "unexpected": []}, context=context)
    assert malformed.parse_errors
