from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from jsonschema import Draft202012Validator

from .trace import ConversationRecorder


class ToolObservationError(ValueError):
    pass


class ToolInfrastructureError(RuntimeError):
    pass


class ToolTaskDeadlineExceeded(BaseException):
    pass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], str]

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def validate_arguments(self, arguments: Any) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            raise ValueError(
                f"invalid_arguments for {self.name}: expected a JSON object, got {type(arguments).__name__}"
            )
        errors = sorted(
            Draft202012Validator(self.parameters).iter_errors(arguments),
            key=lambda error: tuple((str(part) for part in error.absolute_path)),
        )
        if errors:
            error = errors[0]
            location = "$" + "".join(
                (
                    f"[{part}]" if isinstance(part, int) else f".{part}"
                    for part in error.absolute_path
                )
            )
            raise ValueError(
                f"invalid_arguments for {self.name} at {location}: {error.message}"
            )
        return arguments


@dataclass
class ToolEvent:
    step: int
    name: str
    arguments: dict[str, Any]
    output: str
    raw_action: str | None = None
    error: str | None = None
    done: bool = False
    reward: float | None = None


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Any
    raw_action: str | None = None


@dataclass
class ToolTask:
    task_id: str
    task_description: str
    tools: list[ToolSpec]
    submission_tool: str
    max_steps: int = 100
    initial_observation: str = "Task initialized."
    done: bool = False
    final_answer: str | None = None
    reward: float | None = None
    events: list[ToolEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    model_usage: dict[str, int | bool | None] = field(
        default_factory=lambda: {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "generation_tokens": 0,
            "cached_tokens": 0,
            "usage_complete": True,
            "usage_is_lower_bound": False,
        }
    )

    def __post_init__(self) -> None:
        self._conversation_recorder = ConversationRecorder(task_id=self.task_id)

    def record_conversation(self, **kwargs: Any) -> int:
        return self._conversation_recorder.record(**kwargs)

    def extend_conversation_records(self, records: Any) -> None:
        self._conversation_recorder.extend(records)

    @property
    def conversation_records(self) -> list[dict[str, Any]]:
        return self._conversation_recorder.records

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [tool.openai_schema() for tool in self.tools]

    def tool_map(self) -> dict[str, ToolSpec]:
        return {tool.name: tool for tool in self.tools}

    def execute(
        self, name: str, arguments: Any, *, raw_action: str | None = None
    ) -> str:
        if self.done:
            return json.dumps({"error": "task_already_finished"})
        if len(self.events) >= self.max_steps:
            return json.dumps({"error": "external_step_budget_exhausted"})
        tool = self.tool_map().get(name)
        if tool is not None:
            try:
                arguments = tool.validate_arguments(arguments)
            except ValueError as exc:
                return self.reject_action(
                    raw_action
                    or json.dumps(
                        {"tool": name, "arguments": arguments},
                        ensure_ascii=False,
                        default=str,
                    ),
                    str(exc),
                )
        elif not isinstance(arguments, dict):
            return self.reject_action(
                raw_action or str(arguments),
                f"invalid_arguments for {name}: expected a JSON object, got {type(arguments).__name__}",
            )
        if tool is None:
            return self.reject_action(
                raw_action
                or json.dumps(
                    {"tool": name, "arguments": arguments},
                    ensure_ascii=False,
                    default=str,
                ),
                f"unknown_tool: {name}",
            )
        error = None
        try:
            output = str(tool.handler(arguments))
        except ToolObservationError as exc:
            error = f"{type(exc).__name__}: {exc}"
            output = json.dumps({"error": error}, ensure_ascii=False)
        self.events.append(
            ToolEvent(
                step=len(self.events) + 1,
                name=name,
                arguments=arguments,
                output=output,
                raw_action=raw_action,
                error=error,
                done=self.done,
                reward=self.reward,
            )
        )
        return output

    def execute_batch(
        self,
        actions: list[ToolCall | dict[str, Any] | tuple[str, Any]],
        *,
        raw_action: str | None = None,
    ) -> list[str]:
        try:
            calls = [self._coerce_tool_call(action) for action in actions]
            if not calls:
                raise ValueError("tool batch must contain at least one action")
            if self.done:
                raise ValueError("task_already_finished")
            remaining = self.max_steps - len(self.events)
            if len(calls) > remaining:
                raise ValueError(
                    f"tool batch requires {len(calls)} external steps but only {remaining} remain"
                )
            for index, call in enumerate(calls):
                tool = self.tool_map().get(call.name)
                if tool is None:
                    raise ValueError(
                        f"unknown_tool at batch index {index}: {call.name}"
                    )
                try:
                    tool.validate_arguments(call.arguments)
                except ValueError as exc:
                    raise ValueError(f"batch index {index}: {exc}") from exc
                if call.name == self.submission_tool and index != len(calls) - 1:
                    raise ValueError(
                        "submission tool must be the final action in a batch"
                    )
        except ValueError as exc:
            rendered = raw_action or json.dumps(
                [self._render_batch_action(action) for action in actions],
                ensure_ascii=False,
                default=str,
            )
            return [self.reject_action(rendered, str(exc))]
        return [
            self.execute(
                call.name, call.arguments, raw_action=call.raw_action or raw_action
            )
            for call in calls
        ]

    @staticmethod
    def _coerce_tool_call(
        action: ToolCall | dict[str, Any] | tuple[str, Any],
    ) -> ToolCall:
        if isinstance(action, ToolCall):
            return action
        if isinstance(action, tuple) and len(action) == 2:
            return ToolCall(str(action[0]), action[1])
        if isinstance(action, dict):
            name = str(action.get("name") or action.get("tool") or "").strip()
            if not name:
                raise ValueError("tool batch action is missing a tool name")
            return ToolCall(
                name=name,
                arguments=action.get("arguments", action.get("args", {})),
                raw_action=action.get("raw_action"),
            )
        raise ValueError(f"unsupported tool batch action: {type(action).__name__}")

    @staticmethod
    def _render_batch_action(action: Any) -> Any:
        if isinstance(action, ToolCall):
            return {"name": action.name, "arguments": action.arguments}
        return action

    def record_model_usage(
        self,
        *,
        prompt_tokens: int,
        generation_tokens: int,
        cached_tokens: int | None = None,
    ) -> None:
        self.record_model_usage_batch(
            llm_calls=1,
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            cached_tokens=cached_tokens,
            usage_complete=True,
        )

    def record_model_response(
        self,
        *,
        prompt_tokens: int | None,
        generation_tokens: int | None,
        cached_tokens: int | None = None,
    ) -> None:
        complete = isinstance(prompt_tokens, int) and isinstance(generation_tokens, int)
        self.record_model_usage_batch(
            llm_calls=1,
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            cached_tokens=cached_tokens,
            usage_complete=complete,
        )

    def record_model_usage_batch(
        self,
        *,
        llm_calls: int,
        prompt_tokens: int | None,
        generation_tokens: int | None,
        usage_complete: bool,
        cached_tokens: int | None = None,
    ) -> None:
        if (
            not isinstance(llm_calls, int)
            or isinstance(llm_calls, bool)
            or llm_calls < 0
        ):
            raise ValueError("llm_calls must be a non-negative integer")
        token_values = {
            "prompt_tokens": prompt_tokens,
            "generation_tokens": generation_tokens,
            "cached_tokens": cached_tokens,
        }
        for name, value in token_values.items():
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")
        exact = (
            usage_complete
            and prompt_tokens is not None
            and (generation_tokens is not None)
        )
        if usage_complete and (not exact):
            raise ValueError("complete usage requires both token counts")
        prior_calls = self.model_usage.get("llm_calls")
        prior_prompt = self.model_usage.get("prompt_tokens")
        prior_generation = self.model_usage.get("generation_tokens")
        prior_cached = self.model_usage.get("cached_tokens")
        prior_complete = bool(self.model_usage.get("usage_complete"))
        had_prior_responses = isinstance(prior_calls, int) and prior_calls > 0
        cumulative_complete = exact and (prior_complete or not had_prior_responses)
        self.model_usage = {
            "llm_calls": (prior_calls if isinstance(prior_calls, int) else 0)
            + llm_calls,
            "prompt_tokens": (prior_prompt if isinstance(prior_prompt, int) else 0)
            + (prompt_tokens if isinstance(prompt_tokens, int) else 0),
            "generation_tokens": (
                prior_generation if isinstance(prior_generation, int) else 0
            )
            + (generation_tokens if isinstance(generation_tokens, int) else 0),
            "usage_complete": cumulative_complete,
            "cached_tokens": (prior_cached if isinstance(prior_cached, int) else 0)
            + (cached_tokens if isinstance(cached_tokens, int) else 0),
            "usage_is_lower_bound": not cumulative_complete,
        }

    def mark_model_usage_incomplete(self) -> None:
        self.model_usage.update({"usage_complete": False, "usage_is_lower_bound": True})

    def reject_action(self, raw_action: str, error: str) -> str:
        if len(self.events) >= self.max_steps:
            return json.dumps({"error": "external_step_budget_exhausted"})
        output = json.dumps(
            {
                "error": error,
                "expected": {"tool": "tool_name", "arguments": {}},
                "available_tools": sorted(self.tool_map()),
            },
            ensure_ascii=False,
        )
        self.events.append(
            ToolEvent(
                step=len(self.events) + 1,
                name="invalid_action",
                arguments={},
                output=output,
                raw_action=raw_action,
                error=error,
                done=False,
                reward=None,
            )
        )
        return output

    def finish(self, answer: str, *, reward: float | None = None) -> None:
        self.final_answer = answer.strip()
        self.reward = reward
        self.done = True

    @property
    def steps(self) -> int:
        return len(self.events)

    @property
    def exhausted(self) -> bool:
        return self.steps >= self.max_steps and (not self.done)


def string_parameters(**properties: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            name: {"type": "string", "description": description}
            for name, description in properties.items()
        },
        "required": list(properties),
        "additionalProperties": False,
    }
