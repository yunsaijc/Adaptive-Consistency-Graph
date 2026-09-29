from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import dataclass


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class BudgetLimits:
    max_requests: int = 1000
    max_executor_calls: int = 100
    max_graph_calls: int = 100
    max_input_bytes: int = 16000000
    max_output_tokens: int = 500000
    timeout_seconds: float = 1800

    def __post_init__(self):
        for name, value in vars(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive")
            if name != "timeout_seconds" and type(value) is not int:
                raise ValueError(f"{name} must be an integer")


class RequestBudget:
    def __init__(self, limits: BudgetLimits | None = None):
        self.limits = limits or BudgetLimits()
        self.started = time.monotonic()
        self.requests = 0
        self.input_bytes = 0
        self.output_tokens = 0
        self.roles: Counter[str] = Counter()
        self.blocked = False

    def remaining_seconds(self) -> float:
        remaining = self.limits.timeout_seconds - (time.monotonic() - self.started)
        if self.blocked:
            raise BudgetExceeded(
                "Requests are blocked after an authentication or quota failure"
            )
        if remaining <= 0:
            raise BudgetExceeded("Task deadline exceeded")
        return remaining

    def reserve(self, role: str, input_bytes: int, output_tokens: int = 0) -> None:
        self.remaining_seconds()
        if self.requests >= self.limits.max_requests:
            raise BudgetExceeded("Request budget exhausted")
        role_limit = {
            "executor": self.limits.max_executor_calls,
            "graph": self.limits.max_graph_calls,
        }.get(role)
        if role_limit is not None and self.roles[role] >= role_limit:
            raise BudgetExceeded(f"{role} call budget exhausted")
        if self.input_bytes + input_bytes > self.limits.max_input_bytes:
            raise BudgetExceeded("Request input-byte budget exhausted")
        if self.output_tokens + output_tokens > self.limits.max_output_tokens:
            raise BudgetExceeded("Requested output-token budget exhausted")
        self.requests += 1
        self.roles[role] += 1
        self.input_bytes += input_bytes
        self.output_tokens += output_tokens

    def snapshot(self) -> dict:
        return {
            "requests": self.requests,
            "requests_by_role": dict(self.roles),
            "input_bytes": self.input_bytes,
            "requested_output_tokens": self.output_tokens,
            "blocked": self.blocked,
        }
