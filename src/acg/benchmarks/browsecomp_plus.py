from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from acg.runtime.tools import (
    ToolInfrastructureError,
    ToolObservationError,
    ToolSpec,
    ToolTask,
)

from .common import BenchmarkInfrastructureError, EvaluationResult, read_json

TOP_K = 5
SNIPPET_TOKENS = 512


@dataclass(frozen=True)
class BrowseCompPlusRecord:
    query_id: str
    query: str
    answer: str
    gold_docids: tuple[str, ...] = ()
    evidence_docids: tuple[str, ...] = ()


@dataclass(frozen=True)
class BrowseCompPlusJudgeResult:
    correct: bool | None
    extracted_final_answer: str | None
    reasoning: str | None
    confidence: float | None
    raw_response: str
    error: str | None = None


@dataclass(frozen=True)
class BrowseCompPlusJudgeConfig:
    model: str
    base_url: str
    api_key_env: str
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout_seconds: float = 600.0
    extra_body: dict[str, Any] | None = None

    @property
    def api_key(self) -> str:
        value = os.getenv(self.api_key_env, "")
        if not value:
            raise BenchmarkInfrastructureError(
                f"BrowseComp-Plus judge API key is missing from {self.api_key_env}"
            )
        return value


class SearchClient(Protocol):
    def search(self, query: str) -> list[dict[str, object]]: ...


class DocumentClient(Protocol):
    def get_document(self, docid: str) -> str: ...


class MCPBrowseCompPlusClient:
    def __init__(self, url: str, *, timeout_seconds: float = 120.0) -> None:
        self.url = url
        self.timeout_seconds = timeout_seconds
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="browsecomp-mcp"
        )

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    def preflight(self) -> dict[str, Any]:
        return self._run(self._inspect())

    def search(self, query: str) -> list[dict[str, object]]:
        return validate_search_results(
            self._run(self._call("search", {"query": query}))
        )

    def get_document(self, docid: str) -> str:
        value = self._run(self._call("get_document", {"docid": docid}))
        if isinstance(value, Mapping):
            text = value.get("text") or value.get("document")
        else:
            text = value
        if not isinstance(text, str):
            raise BenchmarkInfrastructureError(
                "MCP get_document must return a text document"
            )
        return text

    def _run(self, coroutine: Any) -> Any:
        return self._executor.submit(asyncio.run, coroutine).result(
            timeout=self.timeout_seconds + 30.0
        )

    async def _inspect(self) -> dict[str, Any]:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(
            self.url, timeout=self.timeout_seconds
        ) as streams:
            read_stream, write_stream, _ = streams
            async with ClientSession(read_stream, write_stream) as session:
                initialized = await session.initialize()
                listed = await session.list_tools()
                names = sorted(str(tool.name) for tool in listed.tools)
                if "search" not in names:
                    raise BenchmarkInfrastructureError(
                        "BrowseComp-Plus MCP does not expose search"
                    )
                return {
                    "tool_names": names,
                    "server": {
                        "name": initialized.serverInfo.name,
                        "version": initialized.serverInfo.version,
                    },
                }

    async def _call(self, name: str, arguments: dict[str, str]) -> Any:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(
            self.url, timeout=self.timeout_seconds
        ) as streams:
            read_stream, write_stream, _ = streams
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
                if result.isError:
                    raise BenchmarkInfrastructureError(
                        f"BrowseComp-Plus MCP {name} failed: {result.content}"
                    )
                texts = [
                    item.text
                    for item in result.content
                    if getattr(item, "type", None) == "text"
                ]
                if len(texts) != 1:
                    raise BenchmarkInfrastructureError(
                        f"BrowseComp-Plus MCP {name} must return one JSON text payload"
                    )
                try:
                    return json.loads(texts[0])
                except json.JSONDecodeError:
                    if name == "get_document":
                        return texts[0]
                    raise BenchmarkInfrastructureError(
                        "BrowseComp-Plus search returned invalid JSON"
                    )


class OpenAICompatibleBrowseCompPlusJudge:
    def __init__(self, config: BrowseCompPlusJudgeConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise BenchmarkInfrastructureError(
                "BrowseComp-Plus judging requires the openai package"
            ) from exc
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=0,
        )

    def __call__(self, question: str, answer: str, response: str) -> str:
        prompt = OFFICIAL_GRADER_TEMPLATE.format(
            question=question, correct_answer=answer, response=response
        )
        completion = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            extra_body=self.config.extra_body,
        )
        return str(completion.choices[0].message.content or "")

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


def load_browsecomp_plus_record(
    dataset_path: str | Path, query_id: str
) -> BrowseCompPlusRecord:
    path = Path(dataset_path).expanduser().resolve()
    rows = _read_rows(path)
    matches = [
        row
        for row in rows
        if str(row.get("query_id") or row.get("id")) == str(query_id)
    ]
    if len(matches) != 1:
        raise KeyError(
            f"expected one BrowseComp-Plus row for {query_id}, found {len(matches)}"
        )
    row = matches[0]
    query = str(row.get("query") or row.get("question") or "").strip()
    answer = str(row.get("answer") or row.get("correct_answer") or "").strip()
    if not query or not answer:
        raise BenchmarkInfrastructureError(
            f"BrowseComp-Plus row {query_id} lacks query or answer"
        )
    return BrowseCompPlusRecord(
        str(query_id),
        query,
        answer,
        _as_docids(row.get("gold_docids")),
        _as_docids(row.get("evidence_docids")),
    )


def build_browsecomp_plus_task(
    record: BrowseCompPlusRecord,
    *,
    search_client: SearchClient,
    max_steps: int = 100,
    allow_document_read: bool = False,
) -> ToolTask:
    task: ToolTask
    document_reader = getattr(search_client, "get_document", None)
    if allow_document_read and not callable(document_reader):
        raise BenchmarkInfrastructureError(
            "document reads were enabled but the retrieval client does not expose get_document"
        )

    def search(arguments: dict[str, Any]) -> str:
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise ToolObservationError("search query must not be empty")
        try:
            rows = validate_search_results(search_client.search(query))
        except BenchmarkInfrastructureError as exc:
            raise ToolInfrastructureError(str(exc)) from exc
        except Exception as exc:
            raise ToolInfrastructureError(
                f"BrowseComp-Plus search backend failed: {type(exc).__name__}: {exc}"
            ) from exc
        task.metadata["searches"].append(
            {"query": query, "docids": [str(row["docid"]) for row in rows]}
        )
        return json.dumps(rows, ensure_ascii=False)

    def submit(arguments: dict[str, Any]) -> str:
        answer = str(arguments.get("answer") or "").strip()
        if not answer:
            raise ToolObservationError("submitted answer must not be empty")
        explanation = str(arguments.get("explanation") or "").strip()
        confidence = arguments.get("confidence")
        if confidence is not None and not 0.0 <= float(confidence) <= 100.0:
            raise ToolObservationError("confidence must be between 0 and 100")
        response = format_official_response(
            answer, explanation, float(confidence) if confidence is not None else None
        )
        task.metadata["official_response"] = response
        task.metadata["submitted_answer"] = answer
        task.finish(answer, reward=None)
        return json.dumps({"submitted": True}, ensure_ascii=False)

    tools = [
        ToolSpec(
            "search",
            "Search the fixed corpus and return five document snippets.",
            {
                "type": "object",
                "properties": {"query": {"type": "string", "minLength": 1}},
                "required": ["query"],
                "additionalProperties": False,
            },
            search,
        )
    ]
    if allow_document_read:

        def get_document(arguments: dict[str, Any]) -> str:
            docid = str(arguments.get("docid") or "").strip()
            if not docid:
                raise ToolObservationError("docid must not be empty")
            try:
                return str(document_reader(docid))
            except Exception as exc:
                raise ToolInfrastructureError(
                    f"BrowseComp-Plus document backend failed: {type(exc).__name__}: {exc}"
                ) from exc

        tools.append(
            ToolSpec(
                "get_document",
                "Read a full document returned by the fixed corpus.",
                {
                    "type": "object",
                    "properties": {"docid": {"type": "string", "minLength": 1}},
                    "required": ["docid"],
                    "additionalProperties": False,
                },
                get_document,
            )
        )
    tools.append(
        ToolSpec(
            "submit_answer",
            "Submit the exact final answer and end the task.",
            {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "minLength": 1},
                    "explanation": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 100},
                },
                "required": ["answer"],
                "additionalProperties": False,
            },
            submit,
        )
    )
    task = ToolTask(
        task_id=record.query_id,
        task_description=record.query,
        tools=tools,
        submission_tool="submit_answer",
        max_steps=max_steps,
        metadata={
            "benchmark": "browsecomp_plus",
            "query": record.query,
            "answer": record.answer,
            "gold_docids": list(record.gold_docids),
            "evidence_docids": list(record.evidence_docids),
            "searches": [],
            "retrieval_protocol": {"top_k": TOP_K, "snippet_tokens": SNIPPET_TOKENS},
        },
    )
    return task


def evaluate_browsecomp_plus_task(
    task: ToolTask,
    *,
    judge_config: BrowseCompPlusJudgeConfig | Mapping[str, Any] | None = None,
    judge: Callable[[str, str, str], str | Mapping[str, Any]] | None = None,
) -> EvaluationResult:
    response = task.metadata.get("official_response")
    if not isinstance(response, str) or not response:
        task.reward = 0.0
        return EvaluationResult("failure", False, error="answer was not submitted")
    question = str(task.metadata["query"])
    answer = str(task.metadata["answer"])
    owned_judge = None
    try:
        if judge is None:
            config = _judge_config(judge_config)
            if config is None:
                raise BenchmarkInfrastructureError(
                    "BrowseComp-Plus evaluation requires BrowseCompPlusJudgeConfig"
                )
            owned_judge = OpenAICompatibleBrowseCompPlusJudge(config)
            judge = owned_judge
        payload = judge(question, answer, response)
        result = parse_judge_payload(payload)
    except BenchmarkInfrastructureError as exc:
        return EvaluationResult("technical_error", None, error=str(exc))
    except Exception as exc:
        return EvaluationResult(
            "technical_error",
            None,
            error=f"BrowseComp-Plus judge failed: {type(exc).__name__}",
        )
    finally:
        if owned_judge is not None:
            owned_judge.close()
    if result.correct is None:
        return EvaluationResult(
            "technical_error",
            None,
            {"judge": result.raw_response},
            result.error or "judge result is unparsable",
        )
    task.reward = 1.0 if result.correct else 0.0
    return EvaluationResult(
        "success" if result.correct else "failure",
        result.correct,
        {"judge": result.__dict__},
    )


def validate_search_results(value: Any) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) != TOP_K:
        raise BenchmarkInfrastructureError(
            f"search must return exactly {TOP_K} results"
        )
    result = []
    seen = set()
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {"docid", "score", "snippet"}:
            raise BenchmarkInfrastructureError(
                "each search row must contain only docid, score, and snippet"
            )
        docid = str(row["docid"])
        score = float(row["score"])
        snippet = row["snippet"]
        if (
            not docid
            or docid in seen
            or not math.isfinite(score)
            or not isinstance(snippet, str)
        ):
            raise BenchmarkInfrastructureError(
                "search rows have invalid docids, scores, or snippets"
            )
        seen.add(docid)
        result.append({"docid": docid, "score": score, "snippet": snippet})
    return result


def format_official_response(
    answer: str, explanation: str = "", confidence: float | None = None
) -> str:
    lines = []
    if explanation:
        lines.append(f"Explanation: {explanation}")
    lines.append(f"Exact Answer: {answer}")
    if confidence is not None:
        lines.append(f"Confidence: {confidence:g}%")
    return "\n".join(lines)


def parse_judge_payload(payload: str | Mapping[str, Any]) -> BrowseCompPlusJudgeResult:
    if isinstance(payload, Mapping):
        if isinstance(payload.get("correct"), bool):
            return BrowseCompPlusJudgeResult(
                payload["correct"],
                _text_or_none(payload.get("extracted_final_answer")),
                _text_or_none(payload.get("reasoning")),
                _number_or_none(payload.get("confidence")),
                json.dumps(dict(payload), ensure_ascii=False),
            )
        raw = payload.get("raw_response")
        if not isinstance(raw, str):
            return BrowseCompPlusJudgeResult(
                None,
                None,
                None,
                None,
                json.dumps(dict(payload), ensure_ascii=False),
                "judge JSON requires boolean correct or raw_response",
            )
    elif isinstance(payload, str):
        raw = payload
    else:
        return BrowseCompPlusJudgeResult(
            None, None, None, None, repr(payload), "judge output has an invalid type"
        )
    correct = re.search(
        r"^\s*(?:\*\*)?correct(?:\*\*)?\s*:\s*(yes|no)\s*$",
        raw,
        re.IGNORECASE | re.MULTILINE,
    )
    answer = _field(raw, "extracted_final_answer")
    reasoning = _field(raw, "reasoning")
    confidence_raw = _field(raw, "confidence")
    confidence_match = re.search(r"[-+]?\d+(?:\.\d+)?", confidence_raw or "")
    confidence = float(confidence_match.group(0)) if confidence_match else None
    if correct is None:
        return BrowseCompPlusJudgeResult(
            None,
            answer,
            reasoning,
            confidence,
            raw,
            "judge response lacks a final correct: yes/no field",
        )
    return BrowseCompPlusJudgeResult(
        correct.group(1).lower() == "yes", answer, reasoning, confidence, raw
    )


def _field(raw: str, name: str) -> str | None:
    match = re.search(
        rf"(?:^|\n)\s*(?:\*\*)?{re.escape(name)}(?:\*\*)?\s*:\s*(.*?)(?=\n\s*(?:\*\*)?(?:extracted_final_answer|reasoning|correct|confidence)(?:\*\*)?\s*:|\Z)",
        raw,
        re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else None


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    value = (
        read_json(path)
        if path.suffix.lower() == ".json"
        else [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    )
    if isinstance(value, Mapping):
        value = value.get("data") or value.get("records")
    if not isinstance(value, list) or not all(
        isinstance(row, Mapping) for row in value
    ):
        raise BenchmarkInfrastructureError(
            f"BrowseComp-Plus dataset is not a list of objects: {path}"
        )
    return [dict(row) for row in value]


def _as_docids(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [value]
    return tuple(str(item) for item in value) if isinstance(value, list) else ()


def _text_or_none(value: Any) -> str | None:
    return str(value) if isinstance(value, str) and value.strip() else None


def _number_or_none(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _judge_config(
    value: BrowseCompPlusJudgeConfig | Mapping[str, Any] | None,
) -> BrowseCompPlusJudgeConfig | None:
    if value is None or isinstance(value, BrowseCompPlusJudgeConfig):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("judge_config must be a mapping or BrowseCompPlusJudgeConfig")
    return BrowseCompPlusJudgeConfig(
        model=str(value["model"]),
        base_url=str(value["base_url"]),
        api_key_env=str(value["api_key_env"]),
        temperature=float(value.get("temperature", 0.7)),
        max_tokens=int(value.get("max_tokens", 4096)),
        timeout_seconds=float(value.get("timeout_seconds", 600.0)),
        extra_body=dict(value["extra_body"])
        if isinstance(value.get("extra_body"), Mapping)
        else None,
    )


OFFICIAL_GRADER_TEMPLATE = (
    "Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.\n\n"
    "[question]: {question}\n\n[response]: {response}\n\n[correct_answer]: {correct_answer}\n\n"
    "Your judgement must be in the format and criteria specified below:\n\n"
    "extracted_final_answer: The final exact answer extracted from the [response].\n\n"
    "[correct_answer]: Repeat the [correct_answer] given above.\n\n"
    "reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.\n\n"
    "correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.\n\n\n"
    "confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.\n\n"
    "Output constraints:\n"
    "- Return only the labeled fields above; do not include markdown fences or extra fields.\n"
    "- Keep extracted_final_answer to the shortest exact answer. Never copy raw responses,\n"
    "  retrieved snippets, citations, or large JSON fields from [response] into it.\n"
    "- Keep reasoning under 120 words and always output a final correct: yes or\n"
    "  correct: no line followed by confidence:.\n"
    "- Keep the complete judgement under 400 words.\n"
)
