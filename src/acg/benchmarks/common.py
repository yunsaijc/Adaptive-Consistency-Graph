from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


class BenchmarkInfrastructureError(RuntimeError):
    pass


@dataclass(frozen=True)
class EvaluationResult:
    status: str
    success: bool | None
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BenchmarkInfrastructureError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkInfrastructureError(f"invalid JSON at {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def clipped(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[truncated to {limit} characters]"


def run_command(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: float | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=dict(env) if env else None,
            check=False,
        )
    except FileNotFoundError as exc:
        raise BenchmarkInfrastructureError(f"command not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise BenchmarkInfrastructureError(
            f"command timed out: {' '.join(command)}"
        ) from exc


@contextmanager
def isolated_import_paths(
    paths: Sequence[Path], modules: Sequence[str]
) -> Iterator[None]:
    before = dict(sys.modules)
    old_path = list(sys.path)
    for module in modules:
        sys.modules.pop(module, None)
    sys.path[:0] = [str(path) for path in paths]
    try:
        yield
    finally:
        sys.path[:] = old_path
        for module in modules:
            sys.modules.pop(module, None)
            if module in before:
                sys.modules[module] = before[module]
