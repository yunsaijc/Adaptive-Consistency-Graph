from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from acg.benchmarks import swe_bench_lite as swe


def test_swe_task_runs_only_in_container_and_parses_harness_report(
    tmp_path: Path, monkeypatch
) -> None:
    record = swe.SWEBenchLiteRecord(
        "owner__repo-1", "owner/repo", "abcdef1", "fix issue", "", "1", {}
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[:5] == ["docker", "exec", "-w", "/testbed", "case1"] and command[
            -3:
        ] == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, "abcdef1\n", "")
        if command[-3:] == ["git", "status", "--porcelain=v1"] or "status" in command:
            return subprocess.CompletedProcess(command, 0, "", "")
        if "diff" in command:
            return subprocess.CompletedProcess(
                command, 0, "diff --git a/a.py b/a.py\n", ""
            )
        if command[0] == "python":
            report = Path(kwargs["cwd"]) / "reports" / "run1.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(
                json.dumps(
                    {
                        "submitted_ids": [record.instance_id],
                        "resolved_ids": [record.instance_id],
                        "error_ids": [],
                    }
                ),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(swe, "run_command", fake_run)
    task = swe.build_swe_bench_lite_task(record, container_name="case1")
    observed = json.loads(task.execute("run_command", {"command": "pwd"}))
    assert observed["returncode"] == 0
    task.execute("submit_patch", {"summary": "fix"})
    harness = tmp_path / "harness"
    (harness / "swebench" / "harness").mkdir(parents=True)
    (harness / "swebench" / "harness" / "run_evaluation.py").write_text(
        "", encoding="utf-8"
    )
    result = swe.evaluate_swe_bench_lite_task(
        task,
        dataset_path=tmp_path / "data.parquet",
        harness_root=harness,
        output_dir=tmp_path / "out",
        run_id="run1",
    )
    assert result.status == "success"
    assert any(
        command[:6] == ["docker", "exec", "-w", "/testbed", "case1", "bash"]
        for command in calls
    )


def test_swe_incomplete_report_is_technical_error(tmp_path: Path, monkeypatch) -> None:
    record = swe.SWEBenchLiteRecord(
        "owner__repo-2", "owner/repo", "abcdef1", "fix issue", "", "1", {}
    )

    def fake_run(command, **kwargs):
        if command[:5] == ["docker", "exec", "-w", "/testbed", "case2"] and command[
            -3:
        ] == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, "abcdef1\n", "")
        if "status" in command:
            return subprocess.CompletedProcess(command, 0, "", "")
        if "diff" in command:
            return subprocess.CompletedProcess(
                command, 0, "diff --git a/a.py b/a.py\n", ""
            )
        if command[0] == "python":
            report = Path(kwargs["cwd"]) / "reports" / "run2.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(
                json.dumps({"resolved_ids": [record.instance_id], "error_ids": []}),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(swe, "run_command", fake_run)
    task = swe.build_swe_bench_lite_task(record, container_name="case2")
    task.execute("submit_patch", {"summary": "fix"})
    harness = tmp_path / "harness"
    (harness / "swebench" / "harness").mkdir(parents=True)
    (harness / "swebench" / "harness" / "run_evaluation.py").write_text(
        "", encoding="utf-8"
    )
    result = swe.evaluate_swe_bench_lite_task(
        task,
        dataset_path=tmp_path / "data.parquet",
        harness_root=harness,
        output_dir=tmp_path / "out",
        run_id="run2",
    )
    assert result.status == "technical_error"
    assert result.success is None


@pytest.mark.parametrize(
    "resolved_ids",
    [pytest.param(None, id="missing"), pytest.param("owner__repo-3", id="string")],
)
def test_swe_malformed_resolved_ids_are_technical_error(
    tmp_path: Path, monkeypatch, resolved_ids
) -> None:
    record = swe.SWEBenchLiteRecord(
        "owner__repo-3", "owner/repo", "abcdef1", "fix issue", "", "1", {}
    )

    def fake_run(command, **kwargs):
        if command[:5] == ["docker", "exec", "-w", "/testbed", "case3"] and command[
            -3:
        ] == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, "abcdef1\n", "")
        if "status" in command:
            return subprocess.CompletedProcess(command, 0, "", "")
        if "diff" in command:
            return subprocess.CompletedProcess(
                command, 0, "diff --git a/a.py b/a.py\n", ""
            )
        if command[0] == "python":
            report = Path(kwargs["cwd"]) / "reports" / "run3.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            payload = {"submitted_ids": [record.instance_id], "error_ids": []}
            if resolved_ids is not None:
                payload["resolved_ids"] = resolved_ids
            report.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(swe, "run_command", fake_run)
    task = swe.build_swe_bench_lite_task(record, container_name="case3")
    task.execute("submit_patch", {"summary": "fix"})
    harness = tmp_path / "harness"
    (harness / "swebench" / "harness").mkdir(parents=True)
    (harness / "swebench" / "harness" / "run_evaluation.py").write_text(
        "", encoding="utf-8"
    )
    result = swe.evaluate_swe_bench_lite_task(
        task,
        dataset_path=tmp_path / "data.parquet",
        harness_root=harness,
        output_dir=tmp_path / "out",
        run_id="run3",
    )
    assert result.status == "technical_error"
    assert result.success is None
