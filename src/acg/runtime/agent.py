from __future__ import annotations

import json
import signal
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from acg.clients import StructuredClient, record_usage
from acg.core import ACGConfig, AdaptiveConsistencyGraph, LLMConceptIndexer, SourceKind

from .budget import BudgetExceeded
from .tokenization import TokenCounter
from .tools import ToolInfrastructureError, ToolTask, ToolTaskDeadlineExceeded
from .trace import write_json


@dataclass(frozen=True)
class AgentConfig:
    max_context: int = 32768
    max_output_tokens: int = 4096
    safety_margin: int = 1024
    graph_output_tokens: int = 2048
    projection_tokens: int = 5000
    max_invalid_outputs: int = 100
    tokenizer_path: str | None = None

    def __post_init__(self):
        for key in (
            "max_context",
            "max_output_tokens",
            "graph_output_tokens",
            "projection_tokens",
            "max_invalid_outputs",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.safety_margin < 0 or self.input_token_budget <= 0:
            raise ValueError("Context must exceed output tokens and safety margin")

    @property
    def input_token_budget(self):
        return self.max_context - self.max_output_tokens - self.safety_margin


@dataclass
class RunResult:
    task_id: str
    status: str
    answer: str | None
    steps: int
    usage: dict
    budget: dict
    error: str | None = None
    error_type: str | None = None
    reward: float | None = None
    trace: list[dict] = field(default_factory=list, repr=False)


@contextmanager
def task_deadline(seconds: float):
    enabled = (
        hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )
    if not enabled:
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer[0] > 0:
        yield
        return

    def expire(signum, frame):
        raise ToolTaskDeadlineExceeded("Task deadline exceeded")

    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


class Agent:
    def __init__(self, client, embedding, *, config: AgentConfig | None = None):
        self.client = client
        self.embedding = embedding
        self.config = config or AgentConfig()
        self.graph = None

    def run(
        self,
        task: ToolTask,
        *,
        output_dir: Path | None = None,
        save_trace: bool = False,
    ) -> RunResult:
        if task.done or task.events:
            raise ValueError("A run requires a fresh task")
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=False)
        secrets = tuple(
            x
            for x in (
                getattr(self.client, "secret", ""),
                getattr(getattr(self.embedding, "client", None), "secret", ""),
            )
            if x
        )
        task._conversation_recorder.enabled = save_trace
        task._conversation_recorder.secrets = secrets
        counter = TokenCounter(self.config.tokenizer_path)
        structured = StructuredClient(
            self.client, task, self.config.graph_output_tokens
        )
        self.graph = AdaptiveConsistencyGraph(
            embedding_provider=self.embedding,
            concept_indexer=LLMConceptIndexer(structured),
            config=ACGConfig(projection_tokens=self.config.projection_tokens),
        )
        trace = []
        status, error, error_type = "step_limit", None, None
        try:
            with task_deadline(self.client.budget.remaining_seconds()):
                self.graph.start(task.task_description)
                status = self._execute(task, counter, trace)
        except (Exception, ToolTaskDeadlineExceeded) as exc:
            error_type = type(exc).__name__
            if isinstance(exc, (BudgetExceeded, ToolTaskDeadlineExceeded)):
                status = "budget_exceeded"
            elif isinstance(exc, ToolInfrastructureError):
                status = "environment_error"
            else:
                status = "error"
            error = str(exc)
            failed = getattr(self.graph, "failed_projection", None)
            if output_dir is not None and failed is not None:
                write_json(
                    output_dir / "render_failure.json",
                    {
                        "projection": failed,
                        "error_type": error_type,
                        "budget": self.config.projection_tokens,
                    },
                    secrets,
                )
        result = RunResult(
            task.task_id,
            status,
            task.final_answer,
            task.steps,
            dict(task.model_usage),
            self.client.budget.snapshot(),
            error,
            error_type,
            task.reward,
            trace,
        )
        if output_dir is not None:
            write_json(
                output_dir / "result.json",
                {k: v for k, v in vars(result).items() if k != "trace"},
                secrets,
            )
            if save_trace:
                write_json(
                    output_dir / "trace.json",
                    {
                        "steps": trace,
                        "events": task.events,
                        "conversation": task.conversation_records,
                    },
                    secrets,
                )
                write_json(output_dir / "graph.json", self.graph.snapshot(), secrets)
        return result

    def _execute(self, task, counter, trace):
        system = f"You are a tool-using agent. Use exactly one tool call per turn. Do not expose private reasoning. Finish only by calling {task.submission_tool}."
        turns = []
        invalid = 0
        calls = 0
        while not task.done and not task.exhausted:
            self.client.budget.remaining_seconds()
            schemas = task.tool_schemas()
            latest = turns[-1][0] if turns else (None, None)
            while (
                counter.count(
                    _messages(system, task.task_description, turns, ""), schemas
                )
                > self.config.input_token_budget
            ):
                if not turns:
                    return "context_budget_exceeded"
                turns.pop(0)
            recent = tuple(source for sources, _ in turns for source in sources)
            available = self.config.input_token_budget - counter.count(
                _messages(system, task.task_description, turns, ""), schemas
            )
            while True:
                view = self.graph.decision_view(
                    recent_source_ids=recent,
                    latest_action_source_id=latest[0],
                    latest_observation_source_id=latest[1],
                    available_memory_tokens=available,
                    token_counter=counter.count_text,
                    excluded_source_ids=tuple(x for x in latest if x is not None),
                )
                rendered = getattr(view, "rendered", None)
                context = str(getattr(rendered, "text", "") or "")
                messages = _messages(system, task.task_description, turns, context)
                input_tokens = counter.count(messages, schemas)
                if input_tokens <= self.config.input_token_budget:
                    break
                if available <= 0:
                    return "context_budget_exceeded"
                available = max(
                    0, available - (input_tokens - self.config.input_token_budget)
                )
            task.record_conversation(
                event="request", role="executor", messages=messages, tools=schemas
            )
            response = self.client.chat(
                messages,
                tools=schemas,
                max_tokens=self.config.max_output_tokens,
                role="executor",
            )
            record_usage(task, response)
            task.record_conversation(
                event="response", role="executor", response=response
            )
            choices = response.get("choices") or []
            message = choices[0].get("message", {}) if choices else {}
            calls += 1
            assistant = {
                key: value
                for key, value in message.items()
                if key in ("role", "content", "tool_calls") and value is not None
            }
            assistant["role"] = "assistant"
            native_calls = assistant.get("tool_calls") or []
            if not native_calls:
                output = task.reject_action(
                    json.dumps(assistant, ensure_ascii=False),
                    "assistant returned no tool call",
                )
                assistant = {
                    "role": "assistant",
                    "content": str(
                        message.get("content") or "Invalid tool-call output omitted."
                    ),
                }
                action = json.dumps(assistant, ensure_ascii=False)
                observation = {
                    "event_type": "action_rejection",
                    "tool_name": None,
                    "content": output,
                }
                replies = [{"role": "user", "content": output}]
                invalid += 1
            else:
                batch, parse_errors, ids, names = _parse_batch(native_calls, calls)
                assistant["tool_calls"] = [
                    dict(call, id=call_id) for call, call_id in zip(native_calls, ids)
                ]
                action = assistant
                if parse_errors:
                    output = task.reject_action(
                        json.dumps(assistant, ensure_ascii=False),
                        "; ".join(parse_errors),
                    )
                    outputs = [output] * len(ids)
                else:
                    outputs = task.execute_batch(
                        batch, raw_action=json.dumps(assistant, ensure_ascii=False)
                    )
                    if (
                        len(outputs) == 1
                        and len(ids) > 1
                        and task.events[-1].name == "invalid_action"
                    ):
                        outputs *= len(ids)
                if task.events[-1].name == "invalid_action":
                    invalid += 1
                observation = {
                    "event_type": "tool_observation_batch",
                    "tool_calls": [
                        {"tool_call_id": call_id, "tool_name": name, "content": output}
                        for call_id, name, output in zip(ids, names, outputs)
                    ],
                }
                replies = [
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name or "unknown",
                        "content": output,
                    }
                    for call_id, name, output in zip(ids, names, outputs)
                ]
            action_source = self.graph.record(SourceKind.ACTION, action)
            observation_source = self.graph.record(SourceKind.OBSERVATION, observation)
            update = (
                None
                if task.done
                else self.graph.update((action_source.id, observation_source.id))
            )
            turns.append(
                ((action_source.id, observation_source.id), [assistant, *replies])
            )
            trace.append(
                {
                    "call": calls,
                    "steps": task.steps,
                    "input_tokens": input_tokens,
                    "context": context,
                    "sources": [action_source.id, observation_source.id],
                    "graph_event_id": getattr(update, "event_id", None),
                }
            )
            if invalid >= self.config.max_invalid_outputs and not task.done:
                return "invalid_output_limit"
        return "completed" if task.done else "step_limit"


def _messages(system, description, turns, context):
    result = [
        {"role": "system", "content": system},
        {"role": "user", "content": description},
    ]
    result.extend(message for _, messages in turns for message in messages)
    if context:
        result.append({"role": "user", "content": context})
    return result


def _parse_batch(calls, turn):
    batch, errors, ids, names = [], [], [], []
    used = set()
    for i, call in enumerate(calls):
        call_id = str(call.get("id") or f"call_{turn}_{i}")
        if call_id in used:
            call_id = f"{call_id}_{i}"
        used.add(call_id)
        function = call.get("function") or {}
        name = str(function.get("name") or "")
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except ValueError:
                arguments = None
        if not isinstance(arguments, dict):
            errors.append(f"Invalid argument object for tool {name}")
            arguments = {}
        batch.append({"name": name, "arguments": arguments})
        ids.append(call_id)
        names.append(name)
    return batch, errors, ids, names
