from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from .clients import EmbeddingClient, ModelClient, ModelConfig
from .example import build_example_task, offline_clients
from .runtime.agent import Agent, AgentConfig
from .runtime.budget import BudgetLimits, RequestBudget
from .runtime.trace import redact, write_json


def load_config(path):
    if path is None:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) - {
        "agent",
        "model",
        "embedding",
        "budget",
    }:
        raise ValueError("Config accepts agent, model, embedding and budget objects")
    return value


def create_clients(config):
    budget = RequestBudget(BudgetLimits(**config.get("budget", {})))
    model_options = dict(config.get("model", {}))
    embedding_options = dict(config.get("embedding", {}))
    for options, prefix in (
        (model_options, "ACG"),
        (embedding_options, "ACG_EMBEDDING"),
    ):
        options.setdefault("model", os.environ.get(prefix + "_MODEL", ""))
        options.setdefault("base_url", os.environ.get(prefix + "_BASE_URL", ""))
        options.setdefault("api_key_env", prefix + "_API_KEY")
        if not options["model"] or not options["base_url"]:
            raise ValueError(
                f"Set {prefix}_MODEL and {prefix}_BASE_URL, or provide them in configuration"
            )
    chat_config = ModelConfig(**model_options)
    embedding_config = ModelConfig(**embedding_options)
    return ModelClient(chat_config, budget), EmbeddingClient(
        ModelClient(embedding_config, budget)
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="acg")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo")
    demo.add_argument("--offline", action="store_true")
    run = sub.add_parser("run")
    run.add_argument("--task", required=True, type=Path)
    for command in (demo, run):
        command.add_argument("--config", type=Path)
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--trace", action="store_true")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--run", required=True, type=Path)
    evaluate.add_argument("--config", type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    chat, embedding, cleanup = None, None, None
    try:
        if args.output.exists():
            raise ValueError("Output directory already exists; choose a new directory")
        if args.command == "evaluate":
            return _evaluate(args)
        config = load_config(args.config)
        if getattr(args, "offline", False):
            chat, embedding, _ = offline_clients(
                BudgetLimits(**config.get("budget", {}))
            )
        else:
            chat, embedding = create_clients(config)
        agent_config = AgentConfig(**config.get("agent", {}))
        if args.command == "demo":
            task = build_example_task()
        else:
            task, cleanup = _benchmark_task(args.task)
        result = Agent(chat, embedding, config=agent_config).run(
            task, output_dir=args.output, save_trace=args.trace
        )
        secrets = tuple(x for x in (chat.secret, embedding.client.secret) if x)
        summary = {k: v for k, v in vars(result).items() if k != "trace"}
        write_json(
            args.output / "task.json",
            {
                "task_id": task.task_id,
                "task_description": task.task_description,
                "submission_tool": task.submission_tool,
                "done": task.done,
                "final_answer": task.final_answer,
                "metadata": task.metadata,
            },
            secrets,
        )
        write_json(
            args.output / "config.json",
            {
                "agent": asdict(AgentConfig(**config.get("agent", {}))),
                "budget": asdict(chat.budget.limits),
                "model": {
                    k: v
                    for k, v in asdict(chat.config).items()
                    if k not in ("base_url", "api_key_env")
                },
                "embedding_model": embedding.client.config.model,
            },
            secrets,
        )
        print(json.dumps(redact(summary, secrets), ensure_ascii=False))
        return 0 if result.status == "completed" else 1
    except (ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
        secrets = tuple(
            x
            for x in (
                getattr(chat, "secret", ""),
                getattr(getattr(embedding, "client", None), "secret", ""),
            )
            if x
        )
        print(redact(f"{type(exc).__name__}: {exc}", secrets), file=sys.stderr)
        return 2
    finally:
        for close in (
            getattr(chat, "close", None),
            getattr(getattr(embedding, "client", None), "close", None),
            cleanup,
        ):
            if close is not None:
                try:
                    close()
                except Exception as exc:
                    print(f"Cleanup failed: {type(exc).__name__}", file=sys.stderr)


def _evaluate(args):
    from .benchmarks.browsecomp_plus import evaluate_browsecomp_plus_task
    from .benchmarks.deepplanning import evaluate_deepplanning_task
    from .benchmarks.swe_bench_lite import evaluate_swe_bench_lite_task
    from .runtime.tools import ToolTask

    saved_result = json.loads((args.run / "result.json").read_text(encoding="utf-8"))
    if saved_result["status"] != "completed":
        raise ValueError("Generation did not complete; no evaluator was called")
    record = json.loads((args.run / "task.json").read_text(encoding="utf-8"))
    task = ToolTask(
        task_id=record["task_id"],
        task_description=record["task_description"],
        tools=[],
        submission_tool=record["submission_tool"],
        done=record["done"],
        final_answer=record["final_answer"],
        metadata=record["metadata"],
    )
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    if not isinstance(config, dict):
        raise ValueError("Evaluation configuration must be a JSON object")
    benchmark = task.metadata.get("benchmark")
    evaluators = {
        "deepplanning": evaluate_deepplanning_task,
        "browsecomp_plus": evaluate_browsecomp_plus_task,
        "swe_bench_lite": evaluate_swe_bench_lite_task,
    }
    if benchmark not in evaluators:
        raise ValueError("This run has no supported benchmark evaluator")
    args.output.mkdir(parents=True, exist_ok=False)
    if benchmark == "swe_bench_lite":
        config["output_dir"] = str(args.output / "harness")
    secrets = tuple(
        os.environ.get(item["api_key_env"], "")
        for item in config.values()
        if isinstance(item, dict) and item.get("api_key_env")
    )
    try:
        result = evaluators[benchmark](task, **config)
    except Exception as exc:
        from .benchmarks.common import EvaluationResult

        result = EvaluationResult(
            "technical_error", None, error=f"Evaluator failed: {type(exc).__name__}"
        )
    write_json(args.output / "evaluation.json", result, secrets)
    print(json.dumps(redact(result, secrets), ensure_ascii=False))
    return 0 if result.success is not None else 1


def _benchmark_task(path):
    from .benchmarks import load_task
    from .benchmarks.browsecomp_plus import MCPBrowseCompPlusClient
    from .benchmarks.swe_bench_lite import (
        load_swe_bench_lite_record,
        start_swe_generation_container,
        stop_swe_generation_container,
    )

    value = json.loads(path.read_text(encoding="utf-8"))
    benchmark = value.pop("benchmark")
    task_id = str(value.pop("task_id"))
    cleanup = None
    if benchmark == "browsecomp_plus":
        search = MCPBrowseCompPlusClient(value.pop("retriever_url"))
        try:
            search.preflight()
        except BaseException:
            search.close()
            raise
        value["search_client"] = search
        cleanup = search.close
    if benchmark == "swe_bench_lite" and value.pop("start_container", False):
        record = load_swe_bench_lite_record(value["config_path"], task_id)
        container = value["container_name"]
        docker = value.get("docker", "docker")
        start_swe_generation_container(record, container_name=container, docker=docker)

        def cleanup():
            stop_swe_generation_container(container, docker=docker)

    try:
        return load_task(benchmark, task_id, **value), cleanup
    except BaseException:
        if cleanup:
            cleanup()
        raise
