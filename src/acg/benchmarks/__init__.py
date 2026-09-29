from __future__ import annotations

from pathlib import Path
from typing import Any

from .browsecomp_plus import build_browsecomp_plus_task, load_browsecomp_plus_record
from .deepplanning import build_shopping_task, build_travel_task
from .swe_bench_lite import build_swe_bench_lite_task, load_swe_bench_lite_record


def load_task(
    benchmark: str,
    task_id: str,
    *,
    assets_path: str | Path | None = None,
    workdir: str | Path | None = None,
    config_path: str | Path | None = None,
    **kwargs: Any,
):
    name = benchmark.strip().lower().replace("-", "_")
    if name == "deepplanning":
        suite = str(kwargs.pop("suite", "")).strip().lower()
        if suite not in {"shopping", "travel"}:
            raise ValueError("DeepPlanning requires suite='shopping' or suite='travel'")
        name = f"deepplanning_{suite}"
    if name == "deepplanning_shopping":
        if assets_path is None:
            raise ValueError("assets_path is required for DeepPlanning shopping")
        if workdir is None:
            raise ValueError("workdir is required for DeepPlanning shopping")
        return build_shopping_task(
            assets_path=assets_path, workdir=workdir, sample_id=task_id, **kwargs
        )
    if name == "deepplanning_travel":
        if assets_path is None:
            raise ValueError("assets_path is required for DeepPlanning travel")
        if workdir is None:
            raise ValueError("workdir is required for DeepPlanning travel")
        return build_travel_task(
            assets_path=assets_path, workdir=workdir, sample_id=task_id, **kwargs
        )
    if name == "browsecomp_plus":
        if config_path is None:
            raise ValueError("config_path is required for BrowseComp-Plus")
        record = load_browsecomp_plus_record(config_path, task_id)
        return build_browsecomp_plus_task(record, **kwargs)
    if name == "swe_bench_lite":
        if config_path is None:
            raise ValueError("config_path is required for SWE-bench Lite")
        record = load_swe_bench_lite_record(config_path, task_id)
        return build_swe_bench_lite_task(record, **kwargs)
    raise ValueError(f"unsupported benchmark: {benchmark}")


__all__ = [
    "build_browsecomp_plus_task",
    "build_shopping_task",
    "build_swe_bench_lite_task",
    "build_travel_task",
    "load_task",
]
