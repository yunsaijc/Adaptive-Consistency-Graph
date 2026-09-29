from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from acg.benchmarks import deepplanning, load_task


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _shopping_assets(root: Path, *, success: bool = True) -> None:
    official = root / "official" / "benchmark" / "deepplanning" / "shoppingplanning"
    _write(
        official / "data" / "level_1_query_meta.json",
        json.dumps([{"id": "7", "query": "find an item"}]),
    )
    _write(
        official / "tools" / "shopping_tool_schema.json",
        json.dumps(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "search_products",
                        "description": "search",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                            "additionalProperties": False,
                        },
                    },
                }
            ]
        ),
    )
    _write(official / "tools" / "__init__.py", "")
    _write(
        official / "tools" / "base_shopping_tool.py",
        "import json\nTOOL_REGISTRY={}\nclass ProductTool:\n    def __init__(self,cfg=None): self.cfg=cfg or {}\n    def call(self,value): return json.dumps({'database_path':self.cfg['database_path'],'query':json.loads(value)['query']})\nTOOL_REGISTRY['search_products']=ProductTool\n",
    )
    _write(
        official / "evaluation" / "evaluation_pipeline.py",
        f"def evaluate_single_case(case_dir): return {{'case_score':{1.0 if success else 0.0},'is_completed':{success!r},'case':case_dir.name}}\ndef generate_case_report(result, report_dir): (report_dir/'result.json').write_text(str(result))\n",
    )
    _write(root / "database_level1" / "case_a" / "cart.json", "{}")


def _travel_assets(root: Path) -> None:
    official = root / "official" / "benchmark" / "deepplanning" / "travelplanning"
    _write(
        official / "data" / "travelplanning_query_en.json",
        json.dumps([{"id": "9", "query": "plan trip"}]),
    )
    _write(
        official / "tools" / "tool_schema_en.json",
        json.dumps(
            [
                {
                    "function": {
                        "name": "search_location",
                        "description": "search",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                            "additionalProperties": False,
                        },
                    }
                }
            ]
        ),
    )
    _write(official / "tools" / "__init__.py", "")
    _write(
        official / "tools" / "base_travel_tool.py",
        "import json\nTOOL_REGISTRY={}\nclass LocationTool:\n    def __init__(self,cfg=None): self.cfg=cfg or {}\n    def call(self,value): return json.dumps({'city':json.loads(value)['city'],'database_path':self.cfg['database_path']})\nTOOL_REGISTRY['search_location']=LocationTool\n",
    )
    _write(
        official / "agent" / "prompts.py",
        "def get_format_convert_prompt(language): return 'convert'\n",
    )
    _write(official / "evaluation" / "x.py", "")
    _write(
        root
        / "travel_database_en"
        / "database_en"
        / "id_9"
        / "locations"
        / "locations_coords.csv",
        "city\nA\n",
    )


def test_shopping_task_uses_copied_official_tools_and_evaluator(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    _shopping_assets(assets)
    task = deepplanning.build_shopping_task(
        assets_path=assets,
        level=1,
        case_id="case_a",
        sample_id="7",
        workdir=tmp_path / "work",
    )
    observed = json.loads(task.execute("search_products", {"query": "tea"}))
    assert observed["query"] == "tea"
    assert "case_a" in observed["database_path"]
    task.execute("submit", {"summary": "done"})
    result = deepplanning.evaluate_deepplanning_task(task)
    assert result.status == "success"
    assert result.success is True


def test_travel_task_uses_structured_external_evaluator_contract(
    tmp_path: Path, monkeypatch
) -> None:
    assets = tmp_path / "assets"
    _travel_assets(assets)
    task = deepplanning.build_travel_task(
        assets_path=assets, language="en", sample_id="id_9", workdir=tmp_path / "work"
    )
    observed = json.loads(task.execute("search_location", {"city": "A"}))
    assert observed["city"] == "A"
    task.execute("submit_report", {"report": "<plan>trip</plan>"})

    monkeypatch.setattr(
        deepplanning,
        "_native_travel_evaluation",
        lambda task, report, config: {"metrics": {"case_acc": 1.0}},
    )
    result = deepplanning.evaluate_deepplanning_task(
        task,
        travel_evaluator_config=deepplanning.TravelEvaluatorConfig(
            "judge", "https://example.test", "TEST_TRAVEL_EVALUATOR_KEY"
        ),
    )
    assert result.status == "success"
    assert result.success is True


def test_shopping_without_submission_scores_as_failure(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    _shopping_assets(assets, success=False)
    task = deepplanning.build_shopping_task(
        assets_path=assets,
        level=1,
        case_id="case_a",
        sample_id="7",
        workdir=tmp_path / "work",
    )
    result = deepplanning.evaluate_deepplanning_task(task)
    assert result.status == "failure"
    assert result.success is False


def test_shopping_missing_evaluator_is_technical_error(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    _shopping_assets(assets)
    task = deepplanning.build_shopping_task(
        assets_path=assets,
        level=1,
        case_id="case_a",
        sample_id="7",
        workdir=tmp_path / "work",
    )
    (
        Path(task.metadata["suite_dir"]) / "evaluation" / "evaluation_pipeline.py"
    ).unlink()
    result = deepplanning.evaluate_deepplanning_task(task)
    assert result.status == "technical_error"
    assert result.success is None


def test_travel_conversion_json_matches_official_raw_writer_shape() -> None:
    assert deepplanning._extract_conversion_json('{"days": []}') == {"days": []}
    assert deepplanning._extract_conversion_json('<JSON>{"days": []}</JSON>') == {
        "days": []
    }


@pytest.mark.skipif(
    not os.getenv("ACG_DEEPPLANNING_ASSETS"),
    reason="requires explicit local official DeepPlanning assets",
)
def test_official_shopping_untouched_cart_scores_failure_without_network(
    tmp_path: Path,
) -> None:
    assets = Path(os.environ["ACG_DEEPPLANNING_ASSETS"])
    task = deepplanning.build_shopping_task(
        assets_path=assets,
        level=1,
        case_id="case_1",
        sample_id="1",
        workdir=tmp_path / "work",
    )
    result = deepplanning.evaluate_deepplanning_task(task)
    assert result.status == "failure"
    assert result.success is False


def test_factory_accepts_canonical_deepplanning_name_and_suite(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    _shopping_assets(assets)
    task = load_task(
        "deepplanning",
        "7",
        assets_path=assets,
        workdir=tmp_path / "work",
        suite="shopping",
        level=1,
        case_id="case_a",
    )
    assert task.task_id == "shopping:1:7"


@pytest.mark.skipif(
    not os.getenv("ACG_DEEPPLANNING_ASSETS"),
    reason="requires explicit local official DeepPlanning assets",
)
def test_official_travel_evaluator_loads_offline_and_scores_invalid_plan(
    tmp_path: Path,
) -> None:
    assets = Path(os.environ["ACG_DEEPPLANNING_ASSETS"])
    sample = str(
        json.loads(
            (
                assets
                / "official"
                / "benchmark"
                / "deepplanning"
                / "travelplanning"
                / "data"
                / "travelplanning_query_en.json"
            ).read_text(encoding="utf-8")
        )[0]["id"]
    )
    task = deepplanning.build_travel_task(
        assets_path=assets, language="en", sample_id=sample, workdir=tmp_path / "work"
    )
    suite = Path(task.metadata["suite_dir"])
    result_root = Path(task.metadata["report_dir"]).parent
    converted = result_root / "converted_plans"
    converted.mkdir(parents=True, exist_ok=True)
    (converted / f"id_{sample}_converted.json").write_text("{}", encoding="utf-8")
    evaluator = deepplanning._load_package(
        suite / "evaluation" / "eval_converted.py",
        "acg_travel_test_evaluator",
        suite / "evaluation",
    )
    summary = evaluator.evaluate_plans(
        result_root,
        suite / "data" / "travelplanning_query_en.json",
        suite / "database" / "database_en",
        workers=1,
    )
    assert summary["metrics"]["case_acc"] == 0.0
