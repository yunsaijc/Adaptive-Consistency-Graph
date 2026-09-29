import json

import pytest
from acg.example import build_example_task, offline_clients
from acg.runtime.agent import Agent, AgentConfig
from acg.runtime.budget import BudgetLimits


def test_full_execution_persists_graph_and_trace(tmp_path):
    chat, embedding, service = offline_clients()
    try:
        agent = Agent(chat, embedding)
        output = tmp_path / "run"
        result = agent.run(build_example_task(), output_dir=output, save_trace=True)
        assert result.status == "completed"
        assert result.answer == "36" and result.steps == 3
        assert result.reward == 1.0
        assert result.budget["requests_by_role"]["graph"] >= 3
        graph = json.loads((output / "graph.json").read_text())
        assert graph["memories"] and graph["events"] and graph["requirements"]
        traces = json.loads((output / "trace.json").read_text())
        assert len(traces["events"]) == 3
        assert traces["conversation"]
        executor = [payload for payload in service.requests if payload.get("tools")]
        assert all(
            payload["messages"][:2] == executor[0]["messages"][:2]
            for payload in executor
        )
    finally:
        chat.close()
        embedding.client.close()


def test_default_does_not_write_conversation_or_graph(tmp_path):
    chat, embedding, _ = offline_clients()
    try:
        output = tmp_path / "run"
        result = Agent(chat, embedding).run(build_example_task(), output_dir=output)
        assert result.status == "completed"
        assert sorted(p.name for p in output.iterdir()) == ["result.json"]
    finally:
        chat.close()
        embedding.client.close()


def test_request_exhaustion_is_not_completion(tmp_path):
    chat, embedding, _ = offline_clients(BudgetLimits(max_executor_calls=1))
    try:
        output = tmp_path / "run"
        result = Agent(chat, embedding).run(build_example_task(), output_dir=output)
        assert result.status == "budget_exceeded"
        assert result.answer is None
        assert result.budget["requests_by_role"]["executor"] == 1
        assert (output / "result.json").exists()
    finally:
        chat.close()
        embedding.client.close()


def test_submission_does_not_trigger_another_graph_request(tmp_path):
    chat, embedding, _ = offline_clients(BudgetLimits(max_graph_calls=3))
    try:
        result = Agent(chat, embedding).run(
            build_example_task(), output_dir=tmp_path / "run"
        )
        assert result.status == "completed" and result.answer == "36"
        assert result.budget["requests_by_role"]["graph"] == 3
    finally:
        chat.close()
        embedding.client.close()


def test_existing_output_is_never_overwritten(tmp_path):
    chat, embedding, service = offline_clients()
    try:
        with pytest.raises(FileExistsError):
            Agent(chat, embedding).run(build_example_task(), output_dir=tmp_path)
        assert service.requests == []
    finally:
        chat.close()
        embedding.client.close()


def test_renderer_failure_stops_without_paid_retry_and_saves_projection(tmp_path):
    chat, embedding, _ = offline_clients()
    try:
        output = tmp_path / "run"
        result = Agent(chat, embedding, config=AgentConfig(projection_tokens=1)).run(
            build_example_task(), output_dir=output
        )
        assert result.status == "error"
        assert result.error_type == "RenderBudgetError"
        assert result.budget["requests_by_role"].get("executor", 0) == 0
        failure = json.loads((output / "render_failure.json").read_text())
        assert failure["projection"]["requirements"]
    finally:
        chat.close()
        embedding.client.close()
