import json
import subprocess
import sys

from acg.cli import main


def test_installed_cli_offline_and_existing_output(tmp_path):
    output = tmp_path / "run"
    command = [
        sys.executable,
        "-m",
        "acg",
        "demo",
        "--offline",
        "--output",
        str(output),
    ]
    result = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    saved = json.loads((output / "result.json").read_text())
    assert saved["status"] == "completed" and saved["answer"] == "36"
    assert saved["budget"]["requests_by_role"]["graph"] == 3
    before = (output / "result.json").read_bytes()
    repeated = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True)
    assert repeated.returncode == 2
    assert (output / "result.json").read_bytes() == before


def test_evaluation_refuses_failed_generation(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "result.json").write_text(json.dumps({"status": "error"}))
    output = tmp_path / "evaluation"
    assert main(["evaluate", "--run", str(run), "--output", str(output)]) == 2
    assert not output.exists()


def test_evaluator_exception_persists_technical_error(tmp_path, monkeypatch):
    from acg.benchmarks import deepplanning

    def fail(task, **config):
        raise RuntimeError("private provider response")

    monkeypatch.setattr(deepplanning, "evaluate_deepplanning_task", fail)
    run = tmp_path / "run"
    run.mkdir()
    (run / "result.json").write_text(json.dumps({"status": "completed"}))
    (run / "task.json").write_text(
        json.dumps(
            {
                "task_id": "1",
                "task_description": "task",
                "submission_tool": "submit",
                "done": True,
                "final_answer": "answer",
                "metadata": {"benchmark": "deepplanning"},
            }
        )
    )
    output = tmp_path / "evaluation"
    assert main(["evaluate", "--run", str(run), "--output", str(output)]) == 1
    result = json.loads((output / "evaluation.json").read_text())
    assert result["status"] == "technical_error" and result["success"] is None
    assert "private provider response" not in json.dumps(result)
