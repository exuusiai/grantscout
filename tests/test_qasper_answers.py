import json
from pathlib import Path

from paperscout.config import Settings
from paperscout.evaluation.datasets import ingest_jsonl
from paperscout.evaluation.qasper import (
    expected_answer_type,
    persist_qasper_evaluation,
    run_qasper_answer_evaluation,
    score_qasper_prediction,
)
from paperscout.models.llm import ChatResponse
from paperscout.retrieval.store import CorpusStore


class FakeAnswerClient:
    def chat(self, **kwargs):
        evidence = json.loads(kwargs["messages"][1]["content"])["evidence"]
        return ChatResponse(
            content=json.dumps(
                {
                    "answer_type": "yes_no",
                    "answer": "yes",
                    "evidence_ids": [evidence[0]["evidence_id"], "invented-evidence"],
                }
            ),
            model="fake",
            usage={"prompt_tokens": 10, "completion_tokens": 3},
        )


def test_qasper_detects_structural_yes_no_questions() -> None:
    assert expected_answer_type("Did they use crowdsourcing?") == "yes_no"
    assert expected_answer_type("Is the result significant?") == "yes_no"
    assert expected_answer_type("What model was used?") == "extractive_or_free_form"


def test_qasper_scores_all_answer_contract_fields() -> None:
    scores = score_qasper_prediction(
        {"answer_type": "extractive", "answer": "the retrieval method", "evidence_ids": ["e1"]},
        [{"answer": {"extractive_spans": ["retrieval method"]}}],
        ["e1"],
    )

    assert scores["answer_f1"] == 1.0
    assert scores["answer_exact_match"] is True
    assert scores["answer_type_correct"] is True
    assert scores["type_and_answer_correct"] is True
    assert scores["evidence_f1"] == 1.0


def test_qasper_strict_correctness_requires_matching_type_and_answer() -> None:
    scores = score_qasper_prediction(
        {"answer_type": "free_form", "answer": "yes", "evidence_ids": []},
        [{"answer": {"yes_no": True, "unanswerable": False}}],
        [],
    )

    assert scores["answer_exact_match"] is True
    assert scores["answer_type_correct"] is False
    assert scores["type_and_answer_correct"] is False


def test_qasper_evaluation_generates_and_filters_evidence_ids(tmp_path: Path) -> None:
    corpus_source = tmp_path / "papers.jsonl"
    corpus_source.write_text(
        json.dumps(
            {
                "id": "paper-1",
                "title": "Evidence Study",
                "full_text": {"Results": ["The method improves recall."]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus.sqlite"
    with CorpusStore(corpus) as store:
        ingest_jsonl(corpus_source, store)
        evidence_id = store.search("improves recall", top_k=1)[0].evidence.id
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps(
            {
                "id": "q1",
                "query": "Does the method improve recall?",
                "relevant_paper_ids": ["paper-1"],
                "relevant_evidence_ids": [evidence_id],
                "qasper_answer": [{"answer": {"yes_no": True, "unanswerable": False}}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_qasper_answer_evaluation(
        corpus,
        queries,
        Settings(runs_dir=tmp_path / "runs"),
        mode="lexical",
        model_client=FakeAnswerClient(),
    )

    assert result["answer_f1"] == 1.0
    assert result["metrics_version"] == "2.0"
    assert result["answer_exact_match"] == 1.0
    assert result["answer_type_accuracy"] == 1.0
    assert result["type_and_answer_accuracy"] == 1.0
    assert result["yes_no_accuracy"] == 1.0
    assert result["model_failures"] == 0
    assert result["per_query"][0]["prediction"]["evidence_ids"] == [evidence_id]
    assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 3}
    artifacts = persist_qasper_evaluation(result, tmp_path / "result.json")
    assert Path(artifacts["json"]).is_file()
    assert "Overall answer F1" in Path(artifacts["markdown"]).read_text(encoding="utf-8")


def test_qasper_retrieval_is_limited_to_annotated_paper(tmp_path: Path) -> None:
    source = tmp_path / "papers.jsonl"
    source.write_text(
        "\n".join(
            json.dumps({"id": paper_id, "title": paper_id, "full_text": {"Results": [text]}})
            for paper_id, text in (
                ("target", "The target method improves recall."),
                ("distractor", "The distractor method improves recall."),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus.sqlite"
    with CorpusStore(corpus) as store:
        ingest_jsonl(source, store)
        results = store.search("improves recall", top_k=5, paper_ids={"target"})
    assert results
    assert {item.paper.id for item in results} == {"target"}


def test_qasper_resume_recomputes_current_metrics(tmp_path: Path) -> None:
    corpus_source = tmp_path / "papers.jsonl"
    corpus_source.write_text(
        json.dumps({"id": "paper-1", "title": "Study", "full_text": {"Results": ["Yes."]}})
        + "\n",
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus.sqlite"
    with CorpusStore(corpus) as store:
        ingest_jsonl(corpus_source, store)
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps(
            {
                "id": "q1",
                "query": "Is it supported?",
                "relevant_paper_ids": ["paper-1"],
                "qasper_answer": [{"answer": {"yes_no": True, "unanswerable": False}}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text(
        json.dumps(
            {
                "id": "q1",
                "prediction": {"answer_type": "free_form", "answer": "yes", "evidence_ids": []},
                "answer_f1": 1.0,
                "answer_type_correct": False,
                "gold_answer_types": ["yes_no"],
                "evidence_precision": 0.0,
                "evidence_recall": 0.0,
                "evidence_f1": 0.0,
                "usage": {},
                "latency_seconds": 0.0,
                "error": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = run_qasper_answer_evaluation(
        corpus,
        queries,
        Settings(runs_dir=tmp_path / "runs"),
        mode="lexical",
        model_client=FakeAnswerClient(),
        checkpoint_path=checkpoint,
    )

    assert result["resumed_predictions"] == 1
    assert result["answer_exact_match"] == 1.0
    assert result["type_and_answer_accuracy"] == 0.0
    assert result["yes_no_accuracy"] == 0.0
