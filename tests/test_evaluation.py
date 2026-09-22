import json
from pathlib import Path

from grantscout.evaluation.datasets import ingest_jsonl
from grantscout.evaluation.metrics import evaluate_queries
from grantscout.evaluation.suite import persist_evaluation_suite, run_evaluation_suite
from grantscout.config import Settings
from grantscout.retrieval.store import CorpusStore


def test_jsonl_ingestion_and_retrieval_metrics(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "json-paper",
                "title": "Grounded Retrieval",
                "full_text": {
                    "Methods": ["We introduce grounded retrieval."],
                    "Results": ["Grounded retrieval improves evidence recall."],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with CorpusStore(tmp_path / "corpus.sqlite") as store:
        counts = ingest_jsonl(source, store)
        evidence_id = store.search("evidence recall", top_k=1)[0].evidence.id
        metrics = evaluate_queries(
            store,
            [
                {
                    "query": "evidence recall",
                    "relevant_paper_ids": ["json-paper"],
                    "relevant_evidence_ids": [evidence_id],
                }
            ],
            top_k=3,
        )

    assert counts == {"papers": 1, "evidence_items": 2}
    assert metrics["recall_at_k"] == 1.0
    assert metrics["evidence_recall_at_k"] == 1.0
    assert metrics["mrr"] == 1.0


def test_evaluation_suite_persists_raw_and_markdown_artifacts(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.sqlite"
    source = tmp_path / "records.jsonl"
    queries = tmp_path / "queries.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "suite-paper",
                "title": "Suite Retrieval",
                "full_text": {"Results": ["The method improves evidence recall."]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with CorpusStore(corpus_path) as store:
        ingest_jsonl(source, store)
        evidence_id = store.search("evidence recall", top_k=1)[0].evidence.id
    queries.write_text(
        json.dumps(
            {
                "query": "evidence recall",
                "relevant_paper_ids": ["suite-paper"],
                "relevant_evidence_ids": [evidence_id],
                "conflict_expected": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = run_evaluation_suite(corpus_path, queries, Settings(runs_dir=tmp_path / "runs"))
    artifacts = persist_evaluation_suite(payload, tmp_path / "runs")

    assert set(payload["ablation"]["variants"]) == {
        "full",
        "without_question_decomposition",
        "without_reranker",
        "without_citation_audit",
        "without_conflict_detection",
    }
    assert Path(artifacts["json"]).is_file()
    assert Path(artifacts["markdown"]).is_file()
    markdown = Path(artifacts["markdown"]).read_text(encoding="utf-8")
    assert "| Paper Recall | Evidence Recall | Runs |" in markdown
    assert "## Traceability and Conflict Metrics" in markdown
    full_summary = payload["ablation"]["variants"]["full"]["summary"]
    assert full_summary["citation_coverage"] == 1.0
    assert full_summary["citation_support_precision"] == 1.0
    assert full_summary["unsupported_claim_rate"] == 0.0
    assert full_summary["conflict_labeled_queries"] == 1


def test_unlabeled_queries_are_excluded_from_evidence_average(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "metric-paper",
                "title": "Metric Retrieval",
                "full_text": {"Results": ["The method improves recall."]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with CorpusStore(tmp_path / "corpus.sqlite") as store:
        ingest_jsonl(source, store)
        metrics = evaluate_queries(
            store,
            [
                {"query": "recall", "relevant_paper_ids": ["metric-paper"]},
                {"query": "recall", "relevant_paper_ids": ["metric-paper"], "relevant_evidence_ids": []},
            ],
            top_k=3,
        )

    assert metrics["evidence_labeled_queries"] == 0
    assert metrics["evidence_recall_at_k"] == 0.0
    assert all(item["evidence_recall"] is None for item in metrics["per_query"])
