import json

import httpx
import pytest
from acg.clients import EmbeddingClient, ModelClient, ModelConfig, ProviderError
from acg.runtime.budget import BudgetExceeded, BudgetLimits, RequestBudget
from acg.runtime.trace import redact


def test_auth_failure_stops_further_requests():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(403, json={"error": "quota"})

    budget = RequestBudget()
    client = ModelClient(
        ModelConfig("test", "https://model.invalid/v1"),
        budget,
        transport=httpx.MockTransport(respond),
    )
    with pytest.raises(ProviderError):
        client.chat([], max_tokens=1, role="executor")
    with pytest.raises(BudgetExceeded):
        client.chat([], max_tokens=1, role="executor")
    assert len(requests) == 1
    client.close()


def test_retry_counts_against_request_budget(monkeypatch):
    monkeypatch.setattr("acg.clients.time.sleep", lambda _: None)
    budget = RequestBudget(BudgetLimits(max_requests=1))
    client = ModelClient(
        ModelConfig("test", "https://model.invalid/v1"),
        budget,
        transport=httpx.MockTransport(lambda _: httpx.Response(503)),
    )
    with pytest.raises(BudgetExceeded):
        client.chat([], max_tokens=1, role="executor")
    assert budget.requests == 1
    client.close()


def test_embedding_request_preserves_order_and_prefix():
    def respond(request):
        assert request.url.path == "/v1/embeddings"
        assert json.loads(request.content)["input"] == ["one", "two"]
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0, 1]},
                    {"index": 0, "embedding": [1, 0]},
                ]
            },
        )

    client = ModelClient(
        ModelConfig("embedding", "https://model.invalid/v1"),
        RequestBudget(),
        transport=httpx.MockTransport(respond),
    )
    assert EmbeddingClient(client).embed(["one", "two"]) == ((1.0, 0.0), (0.0, 1.0))
    client.close()


def test_embedding_rejects_missing_indices():
    client = ModelClient(
        ModelConfig("embedding", "https://model.invalid/v1"),
        RequestBudget(),
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"data": []})),
    )
    with pytest.raises(RuntimeError, match="indices"):
        EmbeddingClient(client).embed(["one"])
    client.close()


def test_budget_reservation_precedes_request():
    called = []
    client = ModelClient(
        ModelConfig("test", "https://model.invalid/v1"),
        RequestBudget(BudgetLimits(max_output_tokens=5)),
        transport=httpx.MockTransport(lambda request: called.append(request)),
    )
    with pytest.raises(BudgetExceeded):
        client.chat([], max_tokens=6, role="executor")
    assert not called
    client.close()


def test_redaction_covers_nested_values_and_custom_keys():
    secret = "test-credential-value"
    value = {
        "authorization": secret,
        "nested": {"text": f"value={secret}"},
        "prompt_tokens": 50,
    }
    cleaned = redact(value, (secret,))
    assert secret not in json.dumps(cleaned)
    assert cleaned["prompt_tokens"] == 50


def test_model_config_rejects_credentials_in_url():
    with pytest.raises(ValueError):
        ModelConfig("test", "https://user:password@model.invalid/v1")
