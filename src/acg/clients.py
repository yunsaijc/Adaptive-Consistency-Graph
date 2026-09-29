from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx

from .runtime.budget import RequestBudget


@dataclass(frozen=True)
class ModelConfig:
    model: str
    base_url: str
    api_key_env: str = "ACG_API_KEY"
    temperature: float = 0.0
    timeout_seconds: float = 120.0
    retries: int = 1
    structured_output: str = "json_schema"
    extra_body: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        parsed = urlsplit(self.base_url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "base_url must be an HTTP(S) API endpoint without credentials or query parameters"
            )
        if not self.model.strip():
            raise ValueError("model is required")
        if self.timeout_seconds <= 0 or not 0 <= self.retries <= 3:
            raise ValueError("Invalid timeout or retry limit")
        if self.structured_output not in ("json_schema", "json_object"):
            raise ValueError("structured_output must be json_schema or json_object")
        if set(self.extra_body) & {
            "model",
            "messages",
            "tools",
            "max_tokens",
            "max_completion_tokens",
            "max_output_tokens",
            "n",
            "stream",
            "response_format",
        }:
            raise ValueError(
                "extra_body cannot override request identity, tools, or budgets"
            )


class ProviderError(RuntimeError):
    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"Model service returned HTTP {status_code}")


class ModelClient:
    def __init__(self, config: ModelConfig, budget: RequestBudget, *, transport=None):
        self.config = config
        self.budget = budget
        self.secret = (
            os.environ.get(config.api_key_env, "") if config.api_key_env else ""
        )
        headers = {"Authorization": f"Bearer {self.secret}"} if self.secret else {}
        self.http = httpx.Client(
            base_url=config.base_url.rstrip("/") + "/",
            headers=headers,
            transport=transport,
            follow_redirects=False,
            trust_env=False,
        )

    def close(self):
        self.http.close()

    def post(self, endpoint: str, payload: dict, role: str) -> dict:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        for attempt in range(self.config.retries + 1):
            self.budget.reserve(role, len(body), int(payload.get("max_tokens", 0)))
            try:
                response = self.http.post(
                    endpoint,
                    content=body,
                    headers={"Content-Type": "application/json"},
                    timeout=min(
                        self.config.timeout_seconds, self.budget.remaining_seconds()
                    ),
                )
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt >= self.config.retries:
                    raise RuntimeError("Model service transport failed") from None
            else:
                if response.status_code in (401, 402, 403):
                    self.budget.blocked = True
                    raise ProviderError(response.status_code)
                if response.status_code == 200:
                    try:
                        result = response.json()
                    except ValueError:
                        raise RuntimeError(
                            "Model service returned invalid JSON"
                        ) from None
                    if not isinstance(result, dict):
                        raise RuntimeError("Model service response must be an object")
                    return result
                if (
                    response.status_code not in (408, 429, 500, 502, 503, 504)
                    or attempt >= self.config.retries
                ):
                    raise ProviderError(response.status_code)
            delay = min(2**attempt, 4)
            if self.budget.remaining_seconds() <= delay:
                raise RuntimeError("Insufficient task time for another request")
            time.sleep(delay)
        raise RuntimeError("Model service request failed")

    def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        role: str,
        tools=None,
        schema=None,
        temperature=None,
    ) -> dict:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.config.temperature
            if temperature is None
            else temperature,
            **self.config.extra_body,
        }
        if tools:
            payload.update(tools=tools, tool_choice="auto", parallel_tool_calls=False)
        if schema is not None:
            payload["response_format"] = (
                {"type": "json_object"}
                if self.config.structured_output == "json_object"
                else {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "graph_update",
                        "schema": schema,
                        "strict": True,
                    },
                }
            )
        return self.post("chat/completions", payload, role)


def record_usage(task, response: dict) -> None:
    usage = response.get("usage") or {}
    detail = usage.get("prompt_tokens_details") or {}
    task.record_model_response(
        prompt_tokens=usage.get("prompt_tokens"),
        generation_tokens=usage.get("completion_tokens"),
        cached_tokens=detail.get("cached_tokens", usage.get("prompt_cache_hit_tokens")),
    )


class StructuredClient:
    def __init__(self, client: ModelClient, task, max_tokens: int = 2048):
        self.client = client
        self.task = task
        self.max_tokens = max_tokens
        self.calls = 0
        self.failures = 0
        self.conversation_recorder = task.record_conversation

    def complete_json(
        self, *, system_prompt: str, user_prompt: str, schema: dict, temperature=None
    ) -> str:
        self.calls += 1
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        self.conversation_recorder(event="request", role="graph", messages=messages)
        try:
            response = self.client.chat(
                messages,
                max_tokens=self.max_tokens,
                role="graph",
                schema=schema,
                temperature=temperature,
            )
        except Exception:
            self.failures += 1
            raise
        record_usage(self.task, response)
        self.conversation_recorder(event="response", role="graph", response=response)
        choices = response.get("choices") or []
        content = choices[0].get("message", {}).get("content") if choices else None
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Graph update returned empty content")
        return content


class EmbeddingClient:
    def __init__(self, client: ModelClient):
        self.client = client
        self.dimensions = None

    def embed(self, texts):
        if not texts:
            return ()
        result = []
        for start in range(0, len(texts), 64):
            batch = list(texts[start : start + 64])
            response = self.client.post(
                "embeddings",
                {
                    "model": self.client.config.model,
                    "input": batch,
                    "encoding_format": "float",
                },
                "embedding",
            )
            records = sorted(
                response.get("data") or [], key=lambda x: x.get("index", -1)
            )
            if [x.get("index") for x in records] != list(range(len(batch))):
                raise RuntimeError("Embedding indices do not match the request")
            for record in records:
                vector = tuple(float(x) for x in record["embedding"])
                if not vector or not all(math.isfinite(x) for x in vector):
                    raise RuntimeError("Embedding contains invalid values")
                if self.dimensions is not None and len(vector) != self.dimensions:
                    raise RuntimeError("Embedding dimensions changed")
                self.dimensions = len(vector)
                result.append(vector)
        return tuple(result)
