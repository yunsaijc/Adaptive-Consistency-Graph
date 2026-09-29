from __future__ import annotations

import hashlib
import json
import re

import httpx

from .clients import EmbeddingClient, ModelClient, ModelConfig
from .runtime.budget import BudgetLimits, RequestBudget
from .runtime.tools import ToolObservationError, ToolSpec, ToolTask, string_parameters


def build_example_task() -> ToolTask:
    documents = {
        "order": {"part": "module", "quantity": 3},
        "catalog": {"module": {"unit_price": 12, "currency": "USD"}},
    }
    task = ToolTask(
        "example",
        "Read the order and catalog. Submit the total order price in USD.",
        [],
        "submit",
        max_steps=10,
    )

    def read(args):
        if args["name"] not in documents:
            raise ToolObservationError("Available documents: order, catalog")
        return json.dumps(documents[args["name"]])

    def submit(args):
        amount = args["amount"]
        task.finish(str(amount), reward=float(amount == 36))
        return json.dumps({"submitted": True})

    task.tools = [
        ToolSpec(
            "read_document",
            "Read an order or catalog document.",
            string_parameters(name="Document name: order or catalog"),
            read,
        ),
        ToolSpec(
            "submit",
            "Submit the total price in USD.",
            {
                "type": "object",
                "properties": {"amount": {"type": "number"}},
                "required": ["amount"],
                "additionalProperties": False,
            },
            submit,
        ),
    ]
    return task


class OfflineService:
    def __init__(self):
        self.executor_calls = 0
        self.requests = []

    def respond(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        self.requests.append(payload)
        if request.url.path.endswith("/embeddings"):
            vectors = []
            for i, text in enumerate(payload["input"]):
                digest = hashlib.sha256(text.encode()).digest()
                vectors.append(
                    {
                        "index": i,
                        "embedding": [(value - 127.5) / 127.5 for value in digest[:16]],
                    }
                )
            return httpx.Response(200, json={"data": vectors})
        if payload.get("tools"):
            actions = [
                ("read_document", {"name": "order"}),
                ("read_document", {"name": "catalog"}),
                ("submit", {"amount": 36}),
            ]
            name, args = actions[min(self.executor_calls, len(actions) - 1)]
            self.executor_calls += 1
            message = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call_{self.executor_calls}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ],
            }
        else:
            user = payload["messages"][-1]["content"]
            if "<<<TASK" in user:
                task = user.split("<<<TASK", 1)[1].split("TASK>>>", 1)[0].strip()
                value = {"requirements": [{"exact_span": task, "label": "Order price"}]}
            else:
                value = {
                    "records": [],
                    "action_origins": [],
                    "user_source_decisions": [],
                }
                decoder = json.JSONDecoder()
                objects = []
                for match in re.finditer(r"{", user):
                    try:
                        obj, _ = decoder.raw_decode(user[match.start() :])
                    except ValueError:
                        continue
                    if isinstance(obj, dict):
                        objects.append(obj)
                for obj in objects:
                    for record in obj.get("current_source_records", []):
                        units = record.get("units", [])
                        value["records"].append(
                            {
                                "record_id": record["record_id"],
                                "existing_concept_ids": [],
                                "new_anchors": [[units[0]["unit_id"]]] if units else [],
                            }
                        )
                requirement_ids = next(
                    (
                        [r["id"] for r in obj["requirements"]]
                        for obj in objects
                        if isinstance(obj.get("requirements"), list)
                        and obj["requirements"]
                        and "id" in obj["requirements"][0]
                    ),
                    [],
                )
                seen = set()
                for obj in objects:
                    if "action_source_id" in obj and "tool_call_id" in obj:
                        key = (obj["action_source_id"], obj["tool_call_id"])
                        if key not in seen:
                            value["action_origins"].append(
                                {
                                    "action_source_id": key[0],
                                    "tool_call_id": key[1],
                                    "requirement_ids": requirement_ids,
                                }
                            )
                            seen.add(key)
            message = {"role": "assistant", "content": json.dumps(value)}
        return httpx.Response(
            200, json={"choices": [{"message": message, "finish_reason": "stop"}]}
        )


def offline_clients(limits: BudgetLimits | None = None):
    service = OfflineService()
    budget = RequestBudget(limits)
    transport = httpx.MockTransport(service.respond)
    model = ModelClient(
        ModelConfig(
            "offline-chat", "https://offline.invalid/v1", api_key_env="", retries=0
        ),
        budget,
        transport=transport,
    )
    embeddings = ModelClient(
        ModelConfig(
            "offline-embedding", "https://offline.invalid/v1", api_key_env="", retries=0
        ),
        budget,
        transport=transport,
    )
    return model, EmbeddingClient(embeddings), service
