from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any


def redact(value: Any, secrets: tuple[str, ...] = ()) -> Any:
    if dataclasses.is_dataclass(value):
        value = {
            item.name: getattr(value, item.name) for item in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(k): "[REDACTED]"
            if re.search(
                r"(^|_)(api_key|password|secret|authorization|credential|access_token)($|_)",
                str(k),
                re.I,
            )
            else redact(v, secrets)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [redact(v, secrets) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"(?i)Bearer\s+[^\s,;]+", "Bearer [REDACTED]", value)
        return re.sub(r"\bsk-[A-Za-z0-9_-]{12,}", "[REDACTED]", value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return redact(value.model_dump(), secrets)
    return redact(str(value), secrets)


class ConversationRecorder:
    def __init__(
        self, task_id: str, enabled: bool = False, secrets: tuple[str, ...] = ()
    ):
        self.task_id = task_id
        self.enabled = enabled
        self.secrets = secrets
        self.records: list[dict[str, Any]] = []
        self._index = 0

    def record(self, **kwargs: Any) -> int:
        self._index += 1
        if self.enabled:
            self.records.append(redact({"index": self._index, **kwargs}, self.secrets))
        return self._index

    def extend(self, records: Any) -> None:
        if self.enabled:
            self.records.extend(redact(list(records), self.secrets))


def write_json(path: Path, data: Any, secrets: tuple[str, ...] = ()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(redact(data, secrets), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
