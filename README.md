<h1 align="center">ACG</h1>
<h3 align="center">Adaptive Consistency Graph for Long-Horizon Agents</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2609.32754"><img src="https://img.shields.io/badge/arXiv-2609.32754-B31B1B.svg" alt="arXiv:2609.32754"></a>
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB.svg" alt="Python 3.11+">
</p>

<p align="center">
  Jiecong Wang &nbsp;·&nbsp; Hao Peng &nbsp;·&nbsp; Zhanyi Wang
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.32754">Paper</a> &nbsp;·&nbsp;
  <a href="#main-results">Results</a> &nbsp;·&nbsp;
  <a href="#quick-start">Quick Start</a> &nbsp;·&nbsp;
  <a href="#configuration">Configuration</a> &nbsp;·&nbsp;
  <a href="#benchmarks">Benchmarks</a> &nbsp;·&nbsp;
  <a href="#citation">Citation</a>
</p>

---

## Overview

ACG keeps task requirements, execution evidence, and the current state connected throughout long-horizon agent execution. It incrementally maintains a persistent graph and constructs a temporary, requirement-centered context for each tool decision.

- **Traceable evidence.** Preserve links between observations, actions, and their sources.
- **Bounded decision context.** Select and organize relevant evidence within a configurable context budget.
- **Tool-based execution.** Run ACG with an OpenAI-compatible model service and your own environment.

This repository accompanies [Adaptive Consistency Graph for Long-Horizon Agents](https://arxiv.org/abs/2609.32754). It includes the ACG implementation, a self-contained offline example, and integrations for DeepPlanning, BrowseComp-Plus, and SWE-bench Lite.

## Main Results

Task success rates (%) from **Table 1** of the [paper](https://arxiv.org/abs/2609.32754v1). All methods use the same task set within each benchmark. Values after ± are bootstrap standard deviations; **Avg.** is the equal-weight arithmetic mean across the three benchmarks, not a pooled task success rate. Bold indicates the best result within each model group.

| Model | Method | DeepPlanning | BrowseComp-Plus | SWE-bench Lite | Avg. |
| :--- | :--- | ---: | ---: | ---: | ---: |
| GPT-5.6-luna | ReAct | 10.8 ± 1.64 | 62.4 ± 1.71 | 60.3 ± 2.84 | 44.5 |
| GPT-5.6-luna | COMPASS | 6.4 ± 1.28 | 48.1 ± 1.72 | 12.7 ± 1.93 | 22.4 |
| GPT-5.6-luna | TDP* | 6.9 ± 1.36 | 10.5 ± 1.05 | 22.3 ± 2.39 | 13.3 |
| GPT-5.6-luna | CUGA | 0.0 ± 0.00 | 10.96 ± 1.08 | 3.67 ± 1.08 | 4.88 |
| GPT-5.6-luna | ACON | 7.8 ± 1.42 | 10.2 ± 1.04 | 50.0 ± 2.89 | 22.7 |
| GPT-5.6-luna | **ACG (Ours)** | **12.4 ± 1.75** | **73.5 ± 1.87** | **64.7 ± 2.76** | **50.2** |
| DeepSeek-v4-flash | ReAct | 6.4 ± 1.28 | **68.2 ± 1.59** | 56.7 ± 2.84 | 43.8 |
| DeepSeek-v4-flash | COMPASS | 10.6 ± 1.62 | 8.7 ± 0.99 | 4.0 ± 1.12 | 7.74 |
| DeepSeek-v4-flash | TDP* | 2.8 ± 0.88 | 18.1 ± 1.34 | 10.7 ± 1.79 | 10.5 |
| DeepSeek-v4-flash | CUGA | 0.0 ± 0.00 | 8.6 ± 0.98 | 3.7 ± 1.07 | 4.07 |
| DeepSeek-v4-flash | ACON | 3.1 ± 0.91 | 48.4 ± 1.73 | 31.3 ± 2.69 | 27.6 |
| DeepSeek-v4-flash | **ACG (Ours)** | **11.7 ± 1.70** | 62.7 ± 1.64 | **63.3 ± 2.76** | **45.9** |

Results are reproduced from the paper; they are not measurements from the offline demo.

## Quick Start

Requires **Python 3.11+**. Run the following from the repository root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
acg --help
acg demo --offline --output outputs/offline
```

The offline demo uses deterministic model responses and embeddings with no network calls or API keys. It reads an order and a catalog, then submits the total price: **36**. Every run requires a new output directory.

To run the test suite:

```sh
python -m pip install -e '.[test]'
pytest -q
```

## Configuration

### Model services

Set these environment variables, or provide the model names and endpoints in a JSON configuration:

| Service | Endpoint | Model | Credential |
| :--- | :--- | :--- | :--- |
| Chat completions | `ACG_BASE_URL` | `ACG_MODEL` | `ACG_API_KEY` |
| Embeddings | `ACG_EMBEDDING_BASE_URL` | `ACG_EMBEDDING_MODEL` | `ACG_EMBEDDING_API_KEY` |

Both endpoints must include the API prefix. Credentials are read from environment variables; [.env.example](.env.example) lists the supported variables. Environment files are not loaded automatically.

The chat service must support native function calls and structured JSON responses. Set `model.structured_output` to `json_object` if JSON Schema output is unavailable; ACG still validates the returned graph data.

```sh
acg demo --config configs/default.json --output outputs/demo --trace
```

### Context and request limits

| Configuration | Purpose |
| :--- | :--- |
| [default.json](configs/default.json) | General runtime settings with explicit request limits |
| [paper.json](configs/paper.json) | Paper-oriented settings: 5,000 graph-context tokens, 2,048 graph-output tokens, 4,096 executor-output tokens, and temperature 0.9 |
| [Task configurations](configs/tasks/) | Benchmark task selection and external asset paths |
| [Evaluation configurations](configs/evaluation/) | Converter, judge, and official harness settings |

Configuration separates context limits, model settings, and request budgets. Input bytes and reserved output tokens are request limits, not monetary estimates. Without a local tokenizer, context budgeting uses UTF-8 byte counts. Install the `tokenizer` extra and set `agent.tokenizer_path` to use a local tokenizer.

Set the model context limit and task timeout for your environment. Matching the paper configuration alone does not reproduce a published score.

### Run outputs

Each CLI run saves `result.json`, `task.json`, and `config.json`. Add `--trace` to save execution and graph records. Outputs may contain task input and tool responses; authentication values are redacted.

## Benchmarks

Benchmark datasets, retrieval indexes, Docker images, and official evaluators are external assets and are not bundled. Install only the integration you need:

| Benchmark | Installation | Required assets |
| :--- | :--- | :--- |
| DeepPlanning | `python -m pip install -e '.[planning]'` | Official tools, evaluator source, and extracted databases |
| BrowseComp-Plus | `python -m pip install -e '.[browse]'` | Dataset and a prepared MCP retriever |
| SWE-bench Lite | `python -m pip install -e '.[swe]'` | Dataset, Docker instance images, and official evaluation harness |

Edit the relevant [task configuration](configs/tasks/) before running. Paths are resolved from the working directory. Use fresh work and output directories for each task.

<details>
<summary><strong>DeepPlanning — shopping and travel</strong></summary>

DeepPlanning expects its official tool/evaluator source and extracted databases in this layout:

```text
assets/deepplanning/
  official/benchmark/deepplanning/shoppingplanning/
  official/benchmark/deepplanning/travelplanning/
  database_level1/case_1/
  travel_database_en/database_en/id_0/
```

Keep the complete official suite directories, including tool schemas and query metadata. The adapter copies the required files into an isolated work directory. Shopping evaluates cart state locally. Travel converts the submitted report with the configured model and passes it to the official evaluator. Set the converter model and endpoint in `configs/evaluation/travel.json` and export `ACG_JUDGE_API_KEY`.

```sh
acg run --task configs/tasks/shopping.json --config configs/paper.json --output outputs/shopping --trace
acg evaluate --run outputs/shopping --output outputs/shopping-evaluation

acg run --task configs/tasks/travel.json --config configs/paper.json --output outputs/travel --trace
acg evaluate --run outputs/travel --config configs/evaluation/travel.json --output outputs/travel-evaluation
```

</details>

<details>
<summary><strong>BrowseComp-Plus — retrieval and answer grading</strong></summary>

BrowseComp-Plus accepts a local JSON, CSV, or parquet dataset. Configure a streamable-HTTP MCP retriever URL whose `search(query)` returns five objects with `docid`, `score`, and `snippet` (512-token snippets). The retriever corpus, index, embedding model and snippet tokenizer must match the intended evaluation. The agent exposes search and answer submission. Grading uses the configured model and official answer-comparison prompt; fill `configs/evaluation/browse.json` and export `ACG_JUDGE_API_KEY`.

```sh
acg run --task configs/tasks/browse.json --config configs/paper.json --output outputs/browse --trace
acg evaluate --run outputs/browse --config configs/evaluation/browse.json --output outputs/browse-evaluation
```

</details>

<details>
<summary><strong>SWE-bench Lite — patch generation and evaluation</strong></summary>

SWE-bench Lite requires Docker, the official instance image, and a local JSON or parquet dataset. The adapter creates an isolated container with networking disabled, executes tools in `/testbed`, and removes its container after generation. Prepare the corresponding `swebench/sweb.eval.x86_64.<instance>:latest` image before running; `__` in the instance ID becomes `_1776_`. Evaluation uses an independently installed official SWE-bench harness. Set its source root and Python interpreter in `configs/evaluation/swe.json`; the interpreter must have the harness dependencies installed.

```sh
acg run --task configs/tasks/swe.json --config configs/paper.json --output outputs/swe --trace
acg evaluate --run outputs/swe --config configs/evaluation/swe.json --output outputs/swe-evaluation
```

</details>

Evaluation writes `evaluation.json` in a new directory. A valid task failure has `success: false`; an unavailable or failed evaluator has `success: null`. Generation errors do not automatically trigger evaluation or another generation attempt. Converter and judge calls use the explicitly configured service and can incur charges.

## Python API

```python
from acg.cli import create_clients
from acg.example import build_example_task
from acg.runtime.agent import Agent

chat, embedding = create_clients({})
try:
    result = Agent(chat, embedding).run(build_example_task())
    print(result.status, result.answer)
finally:
    chat.close()
    embedding.client.close()
```

Custom environments implement `ToolTask` and `ToolSpec` from `acg.runtime.tools`. The core graph is also available through `acg.core.AdaptiveConsistencyGraph`.

## Citation

If you use ACG in your research, please cite our [paper](https://arxiv.org/abs/2609.32754):

```bibtex
@misc{wang2026adaptiveconsistencygraph,
  title         = {Adaptive Consistency Graph for Long-Horizon Agents},
  author        = {Jiecong Wang and Hao Peng and Zhanyi Wang},
  year          = {2026},
  eprint        = {2609.32754},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2609.32754}
}
```
