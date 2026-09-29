from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from acg.runtime.tools import ToolObservationError, ToolSpec, ToolTask

from .common import (
    BenchmarkInfrastructureError,
    EvaluationResult,
    isolated_import_paths,
    read_json,
    write_json,
)


@dataclass(frozen=True)
class DeepPlanningTaskPaths:
    suite: str
    suite_dir: Path
    workdir: Path
    case_dir: Path
    query: dict[str, Any]


@dataclass(frozen=True)
class TravelEvaluatorConfig:
    model: str
    base_url: str
    api_key_env: str
    temperature: float = 0.0
    max_tokens: int = 10240
    timeout_seconds: float = 600.0
    extra_body: dict[str, Any] | None = None

    @property
    def api_key(self) -> str:
        value = os.getenv(self.api_key_env, "")
        if not value:
            raise BenchmarkInfrastructureError(
                f"travel evaluator API key is missing from {self.api_key_env}"
            )
        return value


def build_shopping_task(
    *,
    assets_path: str | Path,
    level: int,
    case_id: str,
    sample_id: str,
    workdir: str | Path,
    max_steps: int = 100,
) -> ToolTask:
    assets = Path(assets_path).expanduser().resolve()
    official = assets / "official" / "benchmark" / "deepplanning" / "shoppingplanning"
    case_source = assets / f"database_level{level}" / case_id
    query_path = official / "data" / f"level_{level}_query_meta.json"
    query_rows = read_json(query_path)
    query = _select_query(query_rows, sample_id)
    if not case_source.is_dir():
        raise FileNotFoundError(f"shopping case directory is missing: {case_source}")
    root = Path(workdir).expanduser().resolve()
    _fresh_workdir(root)
    suite_dir = root / "shoppingplanning"
    _copy_tree(official / "tools", suite_dir / "tools")
    _copy_tree(official / "evaluation", suite_dir / "evaluation")
    case_dir = suite_dir / "database" / case_id
    _copy_tree(case_source, case_dir)
    write_json(suite_dir / "data" / f"level_{level}_query_meta.json", [query])
    paths = DeepPlanningTaskPaths("shopping", suite_dir, root, case_dir, query)
    tools = _shopping_tools(paths)
    task = ToolTask(
        task_id=f"shopping:{level}:{sample_id}",
        task_description=str(query.get("query") or query.get("question") or ""),
        tools=tools,
        submission_tool="submit",
        max_steps=max_steps,
        metadata={
            "benchmark": "deepplanning",
            "suite": "shopping",
            "level": level,
            "sample_id": sample_id,
            "case_id": case_id,
            "assets_path": str(assets),
            "workdir": str(root),
            "suite_dir": str(suite_dir),
            "case_dir": str(case_dir),
        },
    )
    _bind_submit(task, "The cart state is the final evaluated state.")
    return task


def build_travel_task(
    *,
    assets_path: str | Path,
    language: str,
    sample_id: str,
    workdir: str | Path,
    max_steps: int = 100,
) -> ToolTask:
    assets = Path(assets_path).expanduser().resolve()
    official = assets / "official" / "benchmark" / "deepplanning" / "travelplanning"
    database = assets / f"travel_database_{language}" / f"database_{language}"
    query_path = official / "data" / f"travelplanning_query_{language}.json"
    query_rows = read_json(query_path)
    raw_id = str(sample_id).replace("id_", "")
    query = _select_query(query_rows, raw_id)
    if not database.is_dir():
        raise FileNotFoundError(f"travel database is missing: {database}")
    root = Path(workdir).expanduser().resolve()
    _fresh_workdir(root)
    suite_dir = root / "travelplanning"
    _copy_tree(official / "tools", suite_dir / "tools")
    _copy_tree(official / "evaluation", suite_dir / "evaluation")
    for name in ("prompts.py",):
        source = official / "agent" / name
        if not source.is_file():
            raise FileNotFoundError(
                f"required official DeepPlanning asset is missing: {source}"
            )
        destination = suite_dir / "agent" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    _copy_tree(database, suite_dir / "database" / f"database_{language}")
    _write_tool_shim(suite_dir)
    write_json(suite_dir / "data" / f"travelplanning_query_{language}.json", [query])
    report_dir = root / "official_travel_result" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = DeepPlanningTaskPaths("travel", suite_dir, root, report_dir, query)
    tools = _travel_tools(paths, language, raw_id)
    task = ToolTask(
        task_id=f"travel:{language}:id_{raw_id}",
        task_description=str(query.get("query") or query.get("question") or ""),
        tools=tools,
        submission_tool="submit_report",
        max_steps=max_steps,
        metadata={
            "benchmark": "deepplanning",
            "suite": "travel",
            "language": language,
            "sample_id": f"id_{raw_id}",
            "assets_path": str(assets),
            "workdir": str(root),
            "suite_dir": str(suite_dir),
            "report_dir": str(report_dir),
        },
    )
    _bind_travel_submit(task, report_dir, raw_id)
    return task


def evaluate_deepplanning_task(
    task: ToolTask,
    *,
    travel_evaluator_config: TravelEvaluatorConfig | Mapping[str, Any] | None = None,
) -> EvaluationResult:
    metadata = task.metadata
    suite = str(metadata.get("suite") or "")
    if suite == "shopping":
        return _evaluate_shopping(task)
    if suite == "travel":
        config = _travel_evaluator_config(travel_evaluator_config)
        return _evaluate_travel(task, travel_evaluator_config=config)
    raise ValueError(f"unsupported DeepPlanning suite: {suite}")


def _shopping_tools(paths: DeepPlanningTaskPaths) -> list[ToolSpec]:
    schemas = _load_schemas(paths.suite_dir / "tools" / "shopping_tool_schema.json")
    instances = _load_instances(
        paths.suite_dir, "base_shopping_tool", {"database_path": str(paths.case_dir)}
    )
    return _make_tools(schemas, instances)


def _travel_tools(
    paths: DeepPlanningTaskPaths, language: str, raw_id: str
) -> list[ToolSpec]:
    schemas = _load_schemas(paths.suite_dir / "tools" / f"tool_schema_{language}.json")
    root = paths.suite_dir / "database" / f"database_{language}" / f"id_{raw_id}"
    mapping = {
        "query_train_info": "trains/trains.csv",
        "query_flight_info": "flights/flights.csv",
        "query_hotel_info": "hotels/hotels.csv",
        "query_attraction_details": "attractions/attractions.csv",
        "recommend_attractions": "attractions/attractions.csv",
        "search_location": "locations/locations_coords.csv",
        "query_road_route_info": "transportation/distance_matrix.csv",
        "recommend_restaurants": "restaurants/restaurants.csv",
        "query_restaurant_details": "restaurants/restaurants.csv",
    }
    instances = _load_instances(
        paths.suite_dir,
        "base_travel_tool",
        {"language": language},
        mapping=mapping,
        data_root=root,
    )
    return _make_tools(schemas, instances)


def _load_schemas(path: Path) -> list[dict[str, Any]]:
    value = read_json(path)
    if not isinstance(value, list):
        raise BenchmarkInfrastructureError(f"tool schema must be a list: {path}")
    schemas = []
    for row in value:
        function = row.get("function") if isinstance(row, Mapping) else None
        source = function if isinstance(function, Mapping) else row
        if not isinstance(source, Mapping) or not isinstance(source.get("name"), str):
            raise BenchmarkInfrastructureError(f"invalid tool schema in {path}")
        parameters = source.get("parameters")
        if not isinstance(parameters, dict):
            raise BenchmarkInfrastructureError(
                f"tool {source['name']} has no JSON schema"
            )
        schemas.append(
            {
                "name": source["name"],
                "description": str(source.get("description") or ""),
                "parameters": parameters,
            }
        )
    return schemas


def _load_instances(
    suite_dir: Path,
    base_module: str,
    base_config: dict[str, Any],
    *,
    mapping: Mapping[str, str] | None = None,
    data_root: Path | None = None,
) -> dict[str, Any]:
    module_names = ["tools", base_module]
    with isolated_import_paths([suite_dir, suite_dir / "tools"], module_names):
        importlib.import_module("tools")
        base = importlib.import_module(base_module)
        registry = getattr(base, "TOOL_REGISTRY", None)
        if not isinstance(registry, Mapping):
            raise BenchmarkInfrastructureError(
                f"{base_module}.TOOL_REGISTRY is missing"
            )
        result = {}
        for name, cls in registry.items():
            config = dict(base_config)
            relative = (mapping or {}).get(str(name))
            if relative and data_root is not None:
                candidate = data_root / relative
                if candidate.is_file():
                    config["database_path"] = str(candidate)
            result[str(name)] = cls(cfg=config)
        return result


def _make_tools(
    schemas: list[dict[str, Any]], instances: Mapping[str, Any]
) -> list[ToolSpec]:
    tools = []
    for schema in schemas:
        name = str(schema["name"])
        instance = instances.get(name)
        if instance is None:
            raise BenchmarkInfrastructureError(
                f"official schema refers to an unregistered tool: {name}"
            )
        tools.append(
            ToolSpec(
                name=name,
                description=str(schema["description"]),
                parameters=dict(schema["parameters"]),
                handler=_handler(instance),
            )
        )
    return tools


def _handler(instance: Any) -> Callable[[dict[str, Any]], str]:
    def execute(arguments: dict[str, Any]) -> str:
        try:
            result = instance.call(json.dumps(arguments, ensure_ascii=False))
        except Exception as exc:
            raise ToolObservationError(f"{type(exc).__name__}: {exc}") from exc
        return (
            result
            if isinstance(result, str)
            else json.dumps(result, ensure_ascii=False)
        )

    return execute


def _bind_submit(task: ToolTask, description: str) -> None:
    def submit(arguments: dict[str, Any]) -> str:
        task.metadata["submission"] = str(arguments.get("summary") or "")
        task.finish(task.metadata["submission"], reward=None)
        return json.dumps({"submitted": True}, ensure_ascii=False)

    task.tools.append(
        ToolSpec(
            "submit",
            description,
            {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": [],
                "additionalProperties": False,
            },
            submit,
        )
    )


def _bind_travel_submit(task: ToolTask, report_dir: Path, raw_id: str) -> None:
    def submit(arguments: dict[str, Any]) -> str:
        report = str(arguments.get("report") or "").strip()
        if not report:
            raise ToolObservationError("report must not be empty")
        path = report_dir / f"id_{raw_id}.txt"
        path.write_text(report, encoding="utf-8")
        task.metadata["report_path"] = str(path)
        task.finish(report, reward=None)
        return json.dumps(
            {"submitted": True, "report_path": str(path)}, ensure_ascii=False
        )

    task.tools.append(
        ToolSpec(
            "submit_report",
            "Submit the complete official TravelPlanning report.",
            {
                "type": "object",
                "properties": {"report": {"type": "string", "minLength": 1}},
                "required": ["report"],
                "additionalProperties": False,
            },
            submit,
        )
    )


def _evaluate_shopping(task: ToolTask) -> EvaluationResult:
    suite_dir = Path(str(task.metadata["suite_dir"]))
    case_dir = Path(str(task.metadata["case_dir"]))
    evaluator_path = suite_dir / "evaluation" / "evaluation_pipeline.py"
    if not evaluator_path.is_file():
        return EvaluationResult(
            "technical_error",
            None,
            error=f"shopping evaluator missing: {evaluator_path}",
        )
    try:
        module = _load_module(
            evaluator_path, "deepplanning_shopping_evaluation", [suite_dir]
        )
        result = module.evaluate_single_case(case_dir)
        report_dir = suite_dir.parent / "official_report"
        report_dir.mkdir(parents=True, exist_ok=True)
        module.generate_case_report(result, report_dir)
    except Exception as exc:
        return EvaluationResult(
            "technical_error",
            None,
            error=f"shopping evaluator failed: {type(exc).__name__}: {exc}",
        )
    if not isinstance(result, Mapping):
        return EvaluationResult(
            "technical_error",
            None,
            error="shopping evaluator returned a non-object result",
        )
    success = result.get("case_score") == 1.0 and result.get("is_completed") is True
    task.reward = 1.0 if success else 0.0
    return EvaluationResult("success" if success else "failure", success, dict(result))


def _evaluate_travel(
    task: ToolTask, *, travel_evaluator_config: TravelEvaluatorConfig | None
) -> EvaluationResult:
    report_path = Path(str(task.metadata.get("report_path") or ""))
    if not report_path.is_file():
        task.reward = 0.0
        return EvaluationResult(
            "failure", False, error="official travel report was not submitted"
        )
    if travel_evaluator_config is None:
        return EvaluationResult(
            "technical_error",
            None,
            error="travel evaluation requires TravelEvaluatorConfig",
        )
    try:
        evaluation = _native_travel_evaluation(
            task, report_path, travel_evaluator_config
        )
    except Exception as exc:
        return EvaluationResult(
            "technical_error",
            None,
            error=f"travel evaluator failed: {type(exc).__name__}",
        )
    metrics = evaluation.get("metrics") if isinstance(evaluation, Mapping) else None
    if not isinstance(metrics, Mapping) or not isinstance(
        metrics.get("case_acc"), (int, float)
    ):
        return EvaluationResult(
            "technical_error",
            None,
            error="official travel evaluator did not return metrics.case_acc",
        )
    success = metrics["case_acc"] == 1.0
    task.reward = 1.0 if success else 0.0
    return EvaluationResult(
        "success" if success else "failure", success, dict(evaluation)
    )


def _native_travel_evaluation(
    task: ToolTask, report_path: Path, config: TravelEvaluatorConfig
) -> Mapping[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise BenchmarkInfrastructureError(
            "travel conversion requires the openai package"
        ) from exc
    suite_dir = Path(str(task.metadata["suite_dir"]))
    language = str(task.metadata["language"])
    result_dir = report_path.parent.parent
    prompts = _load_module(
        suite_dir / "agent" / "prompts.py",
        "acg_deepplanning_travel_prompts",
        [suite_dir / "agent"],
    )
    prompt_factory = getattr(prompts, "get_format_convert_prompt", None)
    if not callable(prompt_factory):
        raise BenchmarkInfrastructureError(
            "official travel conversion prompt is unavailable"
        )
    client = OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout_seconds,
        max_retries=0,
    )
    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": str(prompt_factory(language))},
                {"role": "user", "content": report_path.read_text(encoding="utf-8")},
            ],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            extra_body=config.extra_body,
        )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    content = str(response.choices[0].message.content or "")
    converted = _extract_conversion_json(content)
    converted_path = (
        result_dir / "converted_plans" / f"{report_path.stem}_converted.json"
    )
    write_json(converted_path, converted)
    evaluator = _load_package(
        suite_dir / "evaluation" / "eval_converted.py",
        "acg_deepplanning_travel_evaluation",
        suite_dir / "evaluation",
    )
    summary = evaluator.evaluate_plans(
        result_dir,
        suite_dir / "data" / f"travelplanning_query_{language}.json",
        suite_dir / "database" / f"database_{language}",
        workers=1,
    )
    if not isinstance(summary, Mapping):
        raise BenchmarkInfrastructureError(
            "official travel evaluator returned a non-object summary"
        )
    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        raise BenchmarkInfrastructureError(
            "official travel evaluator summary has no metrics"
        )
    return dict(summary)


def _travel_evaluator_config(
    value: TravelEvaluatorConfig | Mapping[str, Any] | None,
) -> TravelEvaluatorConfig | None:
    if value is None or isinstance(value, TravelEvaluatorConfig):
        return value
    if not isinstance(value, Mapping):
        raise ValueError(
            "travel_evaluator_config must be a mapping or TravelEvaluatorConfig"
        )
    return TravelEvaluatorConfig(
        model=str(value["model"]),
        base_url=str(value["base_url"]),
        api_key_env=str(value["api_key_env"]),
        temperature=float(value.get("temperature", 0.0)),
        max_tokens=int(value.get("max_tokens", 10240)),
        timeout_seconds=float(value.get("timeout_seconds", 600.0)),
        extra_body=dict(value["extra_body"])
        if isinstance(value.get("extra_body"), Mapping)
        else None,
    )


def _extract_conversion_json(content: str) -> Any:
    match = __import__("re").search(r"<JSON>([\s\S]*?)</JSON>", content)
    payload = match.group(1).strip() if match else content.strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise BenchmarkInfrastructureError(
            f"travel conversion response is not JSON: {exc}"
        ) from exc


def _load_module(path: Path, name: str, import_paths: Sequence[Path]) -> Any:
    old_path = list(sys.path)
    sys.path[:0] = [str(item) for item in import_paths]
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise BenchmarkInfrastructureError(f"cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path


def _load_package(path: Path, name: str, package_dir: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(package_dir)]
    )
    if spec is None or spec.loader is None:
        raise BenchmarkInfrastructureError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(name, None)


def _select_query(rows: Any, sample_id: str) -> dict[str, Any]:
    if not isinstance(rows, list):
        raise BenchmarkInfrastructureError(
            "official DeepPlanning query metadata must be a list"
        )
    selected = [
        dict(row)
        for row in rows
        if isinstance(row, Mapping) and str(row.get("id")) == str(sample_id)
    ]
    if len(selected) != 1:
        raise BenchmarkInfrastructureError(
            f"expected one official query for sample_id={sample_id}, found {len(selected)}"
        )
    return selected[0]


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(
            f"required official DeepPlanning asset is missing: {source}"
        )
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite existing workdir content: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def _fresh_workdir(root: Path) -> None:
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"refusing to reuse populated DeepPlanning workdir: {root}"
        )
    root.mkdir(parents=True, exist_ok=True)


def _write_tool_shim(suite_dir: Path) -> None:
    root = suite_dir / "qwen_agent" / "tools"
    root.mkdir(parents=True, exist_ok=True)
    (suite_dir / "qwen_agent" / "__init__.py").write_text("", encoding="utf-8")
    (root / "__init__.py").write_text("", encoding="utf-8")
    (root / "base.py").write_text(
        "import json\nTOOL_REGISTRY = {}\ndef register_tool(name, allow_overwrite=False):\n    def decorate(cls):\n        if name in TOOL_REGISTRY and not allow_overwrite:\n            raise ValueError(f'Tool {name!r} already registered')\n        cls.name = name\n        TOOL_REGISTRY[name] = cls\n        return cls\n    return decorate\nclass BaseTool:\n    name = ''\n    description = ''\n    parameters = {}\n    def __init__(self, cfg=None):\n        self.cfg = cfg or {}\n    def _verify_json_format_args(self, params, strict_json=False):\n        value = json.loads(params) if isinstance(params, str) else params\n        if not isinstance(value, dict):\n            raise ValueError('Parameters must decode to a JSON object')\n        return value\n    def call(self, params, **kwargs):\n        raise NotImplementedError\n",
        encoding="utf-8",
    )
