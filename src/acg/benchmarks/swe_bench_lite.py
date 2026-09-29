from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from acg.runtime.tools import (
    ToolInfrastructureError,
    ToolObservationError,
    ToolSpec,
    ToolTask,
)

from .common import (
    BenchmarkInfrastructureError,
    EvaluationResult,
    clipped,
    read_json,
    run_command,
)


@dataclass(frozen=True)
class SWEBenchLiteRecord:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str
    version: str
    raw: dict[str, Any]


def load_swe_bench_lite_record(
    dataset_path: str | Path, instance_id: str
) -> SWEBenchLiteRecord:
    path = Path(dataset_path).expanduser().resolve()
    rows = _load_rows(path)
    matches = [row for row in rows if str(row.get("instance_id")) == str(instance_id)]
    if len(matches) != 1:
        raise KeyError(
            f"expected one SWE-bench Lite row for {instance_id}, found {len(matches)}"
        )
    row = matches[0]
    required = ["repo", "base_commit", "problem_statement"]
    missing = [key for key in required if not str(row.get(key) or "").strip()]
    if missing:
        raise BenchmarkInfrastructureError(
            f"SWE-bench Lite row {instance_id} lacks {', '.join(missing)}"
        )
    return SWEBenchLiteRecord(
        str(row["instance_id"]),
        str(row["repo"]),
        str(row["base_commit"]),
        str(row["problem_statement"]),
        str(row.get("hints_text") or ""),
        str(row.get("version") or ""),
        row,
    )


def start_swe_generation_container(
    record: SWEBenchLiteRecord, *, container_name: str, docker: str = "docker"
) -> str:
    _validate_container_name(container_name)
    image = official_instance_image(record.instance_id)
    created = run_command(
        [
            docker,
            "create",
            "--network",
            "none",
            "--name",
            container_name,
            image,
            "tail",
            "-f",
            "/dev/null",
        ],
        timeout=120,
    )
    if created.returncode != 0:
        raise BenchmarkInfrastructureError(
            f"could not create official SWE-bench container: {created.stderr.strip()}"
        )
    started = run_command([docker, "start", container_name], timeout=120)
    if started.returncode != 0:
        run_command([docker, "rm", "-f", container_name], timeout=120)
        raise BenchmarkInfrastructureError(
            f"could not start official SWE-bench container: {started.stderr.strip()}"
        )
    return container_name


def stop_swe_generation_container(
    container_name: str, *, docker: str = "docker"
) -> None:
    _validate_container_name(container_name)
    completed = run_command([docker, "rm", "-f", container_name], timeout=120)
    if completed.returncode != 0:
        raise BenchmarkInfrastructureError(
            f"could not remove container {container_name}: {completed.stderr.strip()}"
        )


def build_swe_bench_lite_task(
    record: SWEBenchLiteRecord,
    *,
    container_name: str,
    max_steps: int = 100,
    command_timeout_seconds: int = 300,
    tool_output_chars: int = 16000,
    docker: str = "docker",
) -> ToolTask:
    _validate_container_name(container_name)
    checkout = inspect_swe_generation_checkout(record, container_name, docker=docker)
    base = str(checkout["baseline_commit"])
    task: ToolTask

    def run_repo_command(arguments: dict[str, Any]) -> str:
        command = str(arguments.get("command") or "")
        if not command.strip():
            raise ToolObservationError("command must not be empty")
        return _container_shell(
            docker, container_name, command, command_timeout_seconds, tool_output_chars
        )

    def read_file(arguments: dict[str, Any]) -> str:
        path = _safe_path(str(arguments.get("path") or ""))
        offset = _line_number(arguments.get("offset"), "offset", 0, default=0)
        limit = _line_number(arguments.get("limit"), "limit", 1, default=None)
        return _read_container_file(
            docker, container_name, path, offset, limit, tool_output_chars
        )

    def write_file(arguments: dict[str, Any]) -> str:
        path = _safe_path(str(arguments.get("path") or ""))
        content = str(arguments.get("content") or "")
        _write_container_file(docker, container_name, path, content)
        return json.dumps({"written": path, "chars": len(content)}, ensure_ascii=False)

    def search_files(arguments: dict[str, Any]) -> str:
        query = str(arguments.get("query") or "")
        target = _safe_path(str(arguments.get("path") or "."), allow_root=True)
        if not query:
            raise ToolObservationError("query must not be empty")
        completed = run_command(
            [
                docker,
                "exec",
                "-w",
                "/testbed",
                container_name,
                "git",
                "grep",
                "-n",
                "-e",
                query,
                "--",
                target,
            ],
            timeout=120,
        )
        return clipped(
            completed.stdout or completed.stderr or "No matches.", tool_output_chars
        )

    def submit_patch(arguments: dict[str, Any]) -> str:
        patch = extract_container_patch(container_name, base, docker=docker)
        task.metadata["model_patch"] = patch
        task.metadata["submission_summary"] = str(arguments.get("summary") or "")
        task.finish(patch, reward=None)
        return json.dumps(
            {
                "submitted": True,
                "patch_chars": len(patch),
                "empty_patch": not bool(patch.strip()),
            },
            ensure_ascii=False,
        )

    tools = [
        ToolSpec(
            "run_command",
            "Run a shell command in the official container checkout.",
            {
                "type": "object",
                "properties": {"command": {"type": "string", "minLength": 1}},
                "required": ["command"],
                "additionalProperties": False,
            },
            run_repo_command,
        ),
        ToolSpec(
            "read_file",
            "Read a UTF-8 source file from the official container checkout.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "offset": {
                        "oneOf": [
                            {"type": "integer", "minimum": 0},
                            {"type": "string", "pattern": "^[0-9]+$"},
                        ]
                    },
                    "limit": {
                        "oneOf": [
                            {"type": "integer", "minimum": 1},
                            {"type": "string", "pattern": "^[0-9]+$"},
                        ]
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            read_file,
        ),
        ToolSpec(
            "write_file",
            "Replace a file in the official container checkout.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            write_file,
        ),
        ToolSpec(
            "search_files",
            "Search text in the official container checkout.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "path": {"type": "string"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            search_files,
        ),
        ToolSpec(
            "submit_patch",
            "Capture the current container git diff as the final prediction.",
            {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": [],
                "additionalProperties": False,
            },
            submit_patch,
        ),
    ]
    task = ToolTask(
        task_id=record.instance_id,
        task_description=f"Repository: {record.repo}\nBase commit: {record.base_commit}\n\nIssue:\n{record.problem_statement}",
        tools=tools,
        submission_tool="submit_patch",
        max_steps=max_steps,
        metadata={
            "benchmark": "swe_bench_lite",
            "repo": record.repo,
            "base_commit": record.base_commit,
            "container_base_commit": base,
            "container_checkout": checkout,
            "container_name": container_name,
            "docker": docker,
            "instance_id": record.instance_id,
        },
    )
    return task


def inspect_swe_generation_checkout(
    record: SWEBenchLiteRecord, container_name: str, *, docker: str = "docker"
) -> dict[str, Any]:
    head = _container_output(
        docker, container_name, ["git", "rev-parse", "HEAD"]
    ).strip()
    if not re.fullmatch(r"[0-9a-fA-F]{7,64}", head):
        raise BenchmarkInfrastructureError(
            f"official container {container_name} has no valid git checkout"
        )
    ancestry = run_command(
        [
            docker,
            "exec",
            "-w",
            "/testbed",
            container_name,
            "git",
            "merge-base",
            "--is-ancestor",
            record.base_commit,
            head,
        ],
        timeout=120,
    )
    if ancestry.returncode != 0:
        raise BenchmarkInfrastructureError(
            f"official container checkout is not based on record base commit {record.base_commit}"
        )
    status = _container_output(
        docker,
        container_name,
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
    )
    if status.strip():
        raise BenchmarkInfrastructureError(
            f"official container checkout is dirty before agent execution: {status[:1000]}"
        )
    image_diff = _container_output(
        docker, container_name, ["git", "diff", "--binary", record.base_commit]
    )
    gold_patch = str(record.raw.get("patch") or "").strip()
    if gold_patch and _normalize_patch(gold_patch) in _normalize_patch(image_diff):
        raise BenchmarkInfrastructureError(
            "official container already contains the reference solution patch"
        )
    changed = _container_output(
        docker, container_name, ["git", "diff", "--name-only", record.base_commit]
    ).splitlines()
    return {
        "dataset_base_commit": record.base_commit,
        "official_setup_commit": head,
        "baseline_commit": head,
        "base_is_ancestor": True,
        "checkout_clean": True,
        "image_setup_changed_paths": changed,
        "reference_patch_checked": bool(gold_patch),
        "reference_patch_absent": True,
        "image_setup_matches_dataset_base": head == record.base_commit,
    }


def extract_container_patch(
    container_name: str, base_commit: str, *, docker: str = "docker"
) -> str:
    prepared = run_command(
        [docker, "exec", "-w", "/testbed", container_name, "git", "add", "-N", "."],
        timeout=120,
    )
    if prepared.returncode != 0:
        raise BenchmarkInfrastructureError(
            f"could not prepare container diff: {prepared.stderr.strip()}"
        )
    completed = run_command(
        [
            docker,
            "exec",
            "-w",
            "/testbed",
            container_name,
            "git",
            "-c",
            "core.fileMode=false",
            "diff",
            "--binary",
            base_commit,
        ],
        timeout=120,
    )
    if completed.returncode != 0:
        raise BenchmarkInfrastructureError(
            f"container git diff failed: {completed.stderr.strip()}"
        )
    return completed.stdout


def evaluate_swe_bench_lite_task(
    task: ToolTask,
    *,
    dataset_path: str | Path,
    harness_root: str | Path,
    output_dir: str | Path,
    run_id: str,
    max_workers: int = 1,
    timeout_seconds: int = 1800,
    python: str = "python",
    model_name_or_path: str = "acg",
) -> EvaluationResult:
    patch = task.metadata.get("model_patch")
    if not isinstance(patch, str):
        task.reward = 0.0
        return EvaluationResult("failure", False, error="patch was not submitted")
    root = Path(harness_root).expanduser().resolve()
    harness = root / "swebench" / "harness" / "run_evaluation.py"
    if not harness.is_file():
        return EvaluationResult(
            "technical_error",
            None,
            error=f"official SWE-bench harness is missing: {harness}",
        )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    predictions = output / "predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "instance_id": task.task_id,
                "model_patch": patch,
                "model_name_or_path": model_name_or_path,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    command = [
        python,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        str(Path(dataset_path).expanduser().resolve()),
        "--split",
        "test",
        "--predictions_path",
        str(predictions),
        "--max_workers",
        str(max_workers),
        "--run_id",
        run_id,
        "--timeout",
        str(timeout_seconds),
        "--instance_ids",
        task.task_id,
    ]
    environment = {**os.environ, "PYTHONPATH": str(root)}
    completed = run_command(
        command, cwd=output, timeout=max(timeout_seconds + 1800, 3600), env=environment
    )
    report = _find_official_report(output, run_id)
    if completed.returncode != 0:
        return EvaluationResult(
            "technical_error",
            None,
            error=f"official harness exited {completed.returncode}: {completed.stderr[-1000:]}",
        )
    if report is None:
        return EvaluationResult(
            "technical_error",
            None,
            error=f"official harness did not write a report; exit={completed.returncode}; stderr={completed.stderr[-1000:]}",
        )
    payload = read_json(report)
    if not isinstance(payload, Mapping):
        return EvaluationResult(
            "technical_error", None, error="official SWE-bench report is not an object"
        )
    report_ids: dict[str, list[str]] = {}
    for field in ("submitted_ids", "resolved_ids", "error_ids"):
        values = payload.get(field)
        if not isinstance(values, list):
            return EvaluationResult(
                "technical_error",
                None,
                dict(payload),
                f"official SWE-bench report has no {field} list",
            )
        if any(not isinstance(value, str) or not value for value in values):
            return EvaluationResult(
                "technical_error",
                None,
                dict(payload),
                f"official SWE-bench report has invalid {field} values",
            )
        if len(set(values)) != len(values):
            return EvaluationResult(
                "technical_error",
                None,
                dict(payload),
                f"official SWE-bench report has duplicate {field} values",
            )
        report_ids[field] = values
    submitted = set(report_ids["submitted_ids"])
    resolved_ids = set(report_ids["resolved_ids"])
    error_ids = set(report_ids["error_ids"])
    if task.task_id not in submitted:
        return EvaluationResult(
            "technical_error",
            None,
            dict(payload),
            "official SWE-bench report does not include the submitted instance",
        )
    if resolved_ids & error_ids:
        return EvaluationResult(
            "technical_error",
            None,
            dict(payload),
            "official SWE-bench report has conflicting resolved and error instances",
        )
    if task.task_id in error_ids:
        return EvaluationResult(
            "technical_error",
            None,
            dict(payload),
            "official SWE-bench harness reported an evaluation error",
        )
    resolved = task.task_id in resolved_ids
    task.reward = 1.0 if resolved else 0.0
    return EvaluationResult(
        "success" if resolved else "failure", resolved, dict(payload)
    )


def official_instance_image(instance_id: str) -> str:
    normalized = instance_id.lower().replace("__", "_1776_")
    return f"swebench/sweb.eval.x86_64.{normalized}:latest"


def _container_shell(
    docker: str, container: str, command: str, timeout: int, limit: int
) -> str:
    try:
        completed = run_command(
            [docker, "exec", "-w", "/testbed", container, "bash", "-lc", command],
            timeout=timeout,
        )
    except BenchmarkInfrastructureError as exc:
        raise ToolInfrastructureError(str(exc)) from exc
    return clipped(
        json.dumps(
            {
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
            ensure_ascii=False,
        ),
        limit,
    )


def _container_output(docker: str, container: str, arguments: Sequence[str]) -> str:
    completed = run_command(
        [docker, "exec", "-w", "/testbed", container, *arguments], timeout=120
    )
    if completed.returncode != 0:
        raise BenchmarkInfrastructureError(
            f"container command failed: {completed.stderr.strip()}"
        )
    return completed.stdout


def _read_container_file(
    docker: str,
    container: str,
    path: str,
    offset: int,
    limit: int | None,
    max_chars: int,
) -> str:
    completed = run_command(
        [docker, "exec", "-w", "/testbed", container, "cat", "--", path], timeout=120
    )
    if completed.returncode != 0:
        return json.dumps({"error": "file_not_found", "path": path})
    lines = completed.stdout.splitlines()
    selected = lines[offset : offset + limit if limit is not None else None]
    return clipped(
        "\n".join(
            f"{index:6d}  {line}" for index, line in enumerate(selected, offset + 1)
        ),
        max_chars,
    )


def _write_container_file(docker: str, container: str, path: str, content: str) -> None:
    parent = str(PurePosixPath(path).parent)
    mkdir = run_command(
        [docker, "exec", "-w", "/testbed", container, "mkdir", "-p", "--", parent],
        timeout=120,
    )
    if mkdir.returncode != 0:
        raise ToolInfrastructureError(
            f"cannot create parent directory: {mkdir.stderr.strip()}"
        )
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    try:
        copied = run_command(
            [docker, "cp", str(temporary), f"{container}:/testbed/{path}"], timeout=120
        )
        if copied.returncode != 0:
            raise ToolInfrastructureError(
                f"cannot write {path}: {copied.stderr.strip()}"
            )
    finally:
        temporary.unlink(missing_ok=True)


def _safe_path(value: str, *, allow_root: bool = False) -> str:
    if not value.strip():
        raise ToolObservationError("path must name a file inside /testbed")
    path = PurePosixPath(value)
    if path.is_absolute():
        try:
            path = path.relative_to("/testbed")
        except ValueError as exc:
            raise ToolObservationError(
                "absolute paths must remain inside /testbed"
            ) from exc
    if ".." in path.parts or (path == PurePosixPath(".") and not allow_root):
        raise ToolObservationError("path must remain inside /testbed")
    return str(path)


def _line_number(
    value: Any, name: str, minimum: int, *, default: int | None
) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ToolObservationError(f"{name} must be an integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and value.isascii() and value.isdigit():
        result = int(value)
    else:
        raise ToolObservationError(f"{name} must be an integer")
    if result < minimum:
        raise ToolObservationError(f"{name} must be at least {minimum}")
    return result


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        value = read_json(path)
        if not isinstance(value, list):
            raise BenchmarkInfrastructureError("SWE-bench JSON dataset must be a list")
        return [dict(row) for row in value if isinstance(row, Mapping)]
    try:
        import pandas as pd
    except ImportError as exc:
        raise BenchmarkInfrastructureError(
            "loading SWE-bench parquet requires pandas and a parquet engine"
        ) from exc
    return [dict(row) for row in pd.read_parquet(path).to_dict(orient="records")]


def _find_official_report(output: Path, run_id: str) -> Path | None:
    candidates = []
    for path in output.rglob("*.json"):
        try:
            value = read_json(path)
        except BenchmarkInfrastructureError:
            continue
        if isinstance(value, Mapping) and any(
            key in value for key in ("resolved_ids", "unresolved_ids", "submitted_ids")
        ):
            candidates.append(path)
    preferred = [path for path in candidates if run_id in path.name]
    return max(preferred or candidates, default=None)


def _validate_container_name(value: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError("container_name contains unsupported characters")


def _normalize_patch(value: str) -> str:
    return "\n".join(
        line.rstrip() for line in value.replace("\r\n", "\n").splitlines()
    ).strip()
