from __future__ import annotations

import json
from pathlib import Path

import pytest
from acg.benchmarks.browsecomp_plus import (
    BenchmarkInfrastructureError,
    BrowseCompPlusRecord,
    build_browsecomp_plus_task,
    evaluate_browsecomp_plus_task,
    load_browsecomp_plus_record,
)


class FixtureCorpus:
    def search(self, query: str):
        return [
            {
                "docid": f"d{index}",
                "score": float(6 - index),
                "snippet": f"{query} {index}",
            }
            for index in range(1, 6)
        ]

    def get_document(self, docid: str) -> str:
        return f"full {docid}"


def test_browse_task_search_document_submission_and_judge() -> None:
    record = BrowseCompPlusRecord("q1", "What is the answer?", "forty two")
    task = build_browsecomp_plus_task(
        record, search_client=FixtureCorpus(), allow_document_read=True
    )
    hits = json.loads(task.execute("search", {"query": "answer"}))
    assert len(hits) == 5
    assert task.execute("get_document", {"docid": "d1"}) == "full d1"
    task.execute("submit_answer", {"answer": "42", "explanation": "[d1]"})
    result = evaluate_browsecomp_plus_task(
        task,
        judge=lambda question, answer, response: (
            "extracted_final_answer: 42\nreasoning: equivalent\ncorrect: yes\nconfidence: 100"
        ),
    )
    assert result.status == "success"
    assert result.success is True


def test_browse_loader_requires_selected_record(tmp_path: Path) -> None:
    dataset = tmp_path / "records.json"
    dataset.write_text(
        json.dumps([{"query_id": "q1", "query": "Q", "answer": "A"}]), encoding="utf-8"
    )
    record = load_browsecomp_plus_record(dataset, "q1")
    assert record.answer == "A"


def test_browse_document_reads_fail_fast_without_backend_support() -> None:
    class SearchOnly:
        def search(self, query: str):
            return [
                {"docid": f"d{index}", "score": float(index), "snippet": query}
                for index in range(5)
            ]

    with pytest.raises(BenchmarkInfrastructureError):
        build_browsecomp_plus_task(
            BrowseCompPlusRecord("q", "Q", "A"),
            search_client=SearchOnly(),
            allow_document_read=True,
        )
