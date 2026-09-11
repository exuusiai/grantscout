from __future__ import annotations

import json
import re
import string
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from paperscout.config import Settings
from paperscout.evaluation.datasets import evaluation_records
from paperscout.models.llm import ModelClientError, OpenAICompatibleClient, parse_json_content
from paperscout.retrieval.semantic import SemanticIndex
from paperscout.retrieval.store import CorpusStore


ANSWER_TYPES = {"extractive", "free_form", "yes_no", "unanswerable"}
YES_NO_QUESTION = re.compile(
    r"^(?:do|does|did|is|are|was|were|can|could|has|have|had|will|would|should)\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, reference: str) -> float:
    predicted = _normalize(prediction).split()
    gold = _normalize(reference).split()
    if not predicted or not gold:
        return float(predicted == gold)
    overlap = sum((Counter(predicted) & Counter(gold)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def expected_answer_type(question: str) -> str:
    """Return a structural type constraint when the question makes it unambiguous."""
    return "yes_no" if YES_NO_QUESTION.match(question.strip()) else "extractive_or_free_form"


def _gold_answers(payload: Any) -> list[dict[str, str]]:
    annotations = payload if isinstance(payload, list) else [payload]
    answers: list[dict[str, str]] = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        answer = annotation.get("answer", annotation)
        if not isinstance(answer, dict):
            continue
        if answer.get("unanswerable"):
            answers.append({"answer_type": "unanswerable", "answer": "unanswerable"})
        elif answer.get("yes_no") is not None:
            answers.append(
                {"answer_type": "yes_no", "answer": "yes" if answer["yes_no"] else "no"}
            )
        elif answer.get("extractive_spans"):
            answers.append(
                {
                    "answer_type": "extractive",
                    "answer": " ".join(str(value) for value in answer["extractive_spans"]),
                }
            )
        elif str(answer.get("free_form_answer") or "").strip():
            answers.append(
                {"answer_type": "free_form", "answer": str(answer["free_form_answer"])}
            )
    return answers


def score_qasper_prediction(
    prediction: dict[str, Any], references: Any, gold_evidence_ids: list[str]
) -> dict[str, Any]:
    gold_answers = _gold_answers(references)
    predicted_type = str(prediction.get("answer_type") or "free_form")
    predicted_answer = str(prediction.get("answer") or "")
    predicted_evidence = {str(value) for value in prediction.get("evidence_ids", [])}
    gold_evidence = {str(value) for value in gold_evidence_ids}
    answer_f1 = max(
        (token_f1(predicted_answer, reference["answer"]) for reference in gold_answers),
        default=0.0,
    )
    type_correct = any(reference["answer_type"] == predicted_type for reference in gold_answers)
    evidence_overlap = len(predicted_evidence & gold_evidence)
    evidence_precision = evidence_overlap / max(len(predicted_evidence), 1)
    evidence_recall = evidence_overlap / max(len(gold_evidence), 1)
    evidence_f1 = (
        2 * evidence_precision * evidence_recall / (evidence_precision + evidence_recall)
        if evidence_precision + evidence_recall
        else 0.0
    )
    return {
        "answer_f1": round(answer_f1, 4),
        "answer_type_correct": type_correct,
        "gold_answer_types": sorted({item["answer_type"] for item in gold_answers}),
        "evidence_precision": round(evidence_precision, 4),
        "evidence_recall": round(evidence_recall, 4),
        "evidence_f1": round(evidence_f1, 4),
    }


def _generate_answer(
    client: OpenAICompatibleClient,
    question: str,
    evidence: list[Any],
    settings: Settings,
) -> tuple[dict[str, Any], dict[str, Any]]:
    required_type = expected_answer_type(question)
    response = client.chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the research-paper question using only the supplied evidence. Return "
                    "one JSON object with answer_type (extractive, free_form, yes_no, or "
                    "unanswerable), answer, and evidence_ids. Copy evidence IDs exactly. Use "
                    "unanswerable only when the supplied evidence cannot answer the question. "
                    "When required_answer_type is yes_no, answer_type must be yes_no and answer "
                    "must be exactly yes or no; do not return a sentence. For other answerable "
                    "questions, use extractive for a short span copied from evidence and free_form "
                    "only when synthesis is necessary."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "required_answer_type": required_type,
                        "evidence": [
                            {"evidence_id": item.evidence.id, "text": item.evidence.text}
                            for item in evidence
                        ],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        max_tokens=min(settings.model_max_tokens, 1024),
        temperature=0.0,
        chat_template_kwargs={"enable_thinking": False},
    )
    payload = parse_json_content(response.content)
    if not isinstance(payload, dict):
        raise ModelClientError("Qasper answer generation did not return an object")
    answer_type = str(payload.get("answer_type") or "").lower()
    if answer_type not in ANSWER_TYPES:
        raise ModelClientError(f"Unsupported Qasper answer type: {answer_type}")
    allowed_ids = {item.evidence.id for item in evidence}
    evidence_ids = [
        str(value) for value in payload.get("evidence_ids", []) if str(value) in allowed_ids
    ]
    return {
        "answer_type": answer_type,
        "answer": str(payload.get("answer") or ""),
        "evidence_ids": list(dict.fromkeys(evidence_ids)),
    }, response.usage


def run_qasper_answer_evaluation(
    corpus: Path,
    queries: Path,
    settings: Settings,
    top_k: int = 5,
    mode: str = "semantic",
    limit: int | None = None,
    model_client: OpenAICompatibleClient | None = None,
    workers: int = 4,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    if mode not in {"lexical", "semantic"}:
        raise ValueError("Qasper evaluation mode must be lexical or semantic")
    records = evaluation_records(queries)
    if limit is not None:
        records = records[:limit]
    semantic_index = None
    if mode == "semantic":
        semantic_index = SemanticIndex(settings.vector_index_path, settings.embedding_model)
        semantic_index.load()
    client = model_client or OpenAICompatibleClient(
        settings.model_base_url,
        settings.model_api_key,
        settings.model_name,
        settings.model_timeout_seconds,
    )
    per_query: list[dict[str, Any]] = []
    usage_totals: Counter[str] = Counter()
    failures = 0
    started_at = time.perf_counter()
    completed: dict[str, dict[str, Any]] = {}
    if checkpoint_path is not None and checkpoint_path.exists():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                completed[str(item["id"])] = item
    prepared: list[tuple[dict[str, Any], list[Any]]] = []
    with CorpusStore(corpus) as store:
        for record in records:
            if str(record.get("id")) in completed:
                continue
            question = str(record["query"])
            paper_ids = {str(value) for value in record.get("relevant_paper_ids", [])}
            results = (
                semantic_index.search(question, top_k=top_k, paper_ids=paper_ids or None)
                if semantic_index is not None
                else store.search(question, top_k=top_k, paper_ids=paper_ids or None)
            )
            prepared.append((record, results))

    def generate(item: tuple[dict[str, Any], list[Any]]) -> dict[str, Any]:
        record, results = item
        question = str(record["query"])
        generation_started = time.perf_counter()
        try:
            prediction, usage = _generate_answer(client, question, results, settings)
            error = None
        except (ModelClientError, ValueError) as exception:
            prediction = {
                "answer_type": "unanswerable",
                "answer": "unanswerable",
                "evidence_ids": [],
            }
            usage = {}
            error = str(exception)
        scores = score_qasper_prediction(
            prediction,
            record.get("qasper_answer"),
            [str(value) for value in record.get("relevant_evidence_ids", [])],
        )
        return {
            "id": record.get("id"),
            "query": question,
            "prediction": prediction,
            **scores,
            "usage": usage,
            "latency_seconds": round(time.perf_counter() - generation_started, 4),
            "error": error,
        }

    generated: list[dict[str, Any]] = []
    checkpoint_handle = None
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_handle = checkpoint_path.open("a", encoding="utf-8")
    try:
        with ThreadPoolExecutor(max_workers=max(workers, 1)) as executor:
            for item in executor.map(generate, prepared):
                generated.append(item)
                if checkpoint_handle is not None:
                    checkpoint_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                    checkpoint_handle.flush()
    finally:
        if checkpoint_handle is not None:
            checkpoint_handle.close()
    generated_by_id = {str(item["id"]): item for item in generated}
    per_query = [
        completed.get(str(record.get("id"))) or generated_by_id[str(record.get("id"))]
        for record in records
    ]
    for item in per_query:
        if item["error"]:
            failures += 1
        usage_totals.update(
            {
                key: int(value)
                for key, value in item["usage"].items()
                if isinstance(value, int)
            }
        )

    def mean(key: str, subset: list[dict[str, Any]] | None = None) -> float:
        values = [float(item[key]) for item in (subset or per_query)]
        return round(sum(values) / len(values), 4) if values else 0.0

    by_type: dict[str, dict[str, Any]] = {}
    for answer_type in ANSWER_TYPES:
        subset = [item for item in per_query if answer_type in item["gold_answer_types"]]
        by_type[answer_type] = {"queries": len(subset), "answer_f1": mean("answer_f1", subset) if subset else 0.0}
    yes_no = [item for item in per_query if "yes_no" in item["gold_answer_types"]]
    unanswerable = [item for item in per_query if "unanswerable" in item["gold_answer_types"]]
    return {
        "queries": len(per_query),
        "mode": mode,
        "top_k": top_k,
        "workers": max(workers, 1),
        "answer_f1": mean("answer_f1"),
        "answer_type_accuracy": round(
            sum(bool(item["answer_type_correct"]) for item in per_query) / max(len(per_query), 1), 4
        ),
        "yes_no_accuracy": round(
            sum(item["answer_f1"] == 1.0 for item in yes_no) / max(len(yes_no), 1), 4
        ),
        "unanswerable_accuracy": round(
            sum(item["prediction"]["answer_type"] == "unanswerable" for item in unanswerable)
            / max(len(unanswerable), 1),
            4,
        ),
        "evidence_f1": mean("evidence_f1"),
        "model_failures": failures,
        "resumed_predictions": len(completed),
        "duration_seconds": round(time.perf_counter() - started_at, 3),
        "mean_generation_latency_seconds": mean("latency_seconds"),
        "usage": dict(usage_totals),
        "by_answer_type": by_type,
        "per_query": per_query,
    }


def render_qasper_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Qasper Answer Quality Evaluation",
        "",
        f"- Queries: {payload['queries']}",
        f"- Retrieval mode: `{payload['mode']}`",
        f"- Top K: {payload['top_k']}",
        f"- Model failures: {payload['model_failures']}",
        f"- Duration seconds: {payload['duration_seconds']}",
        "",
        "| Metric | Score |",
        "| --- | ---: |",
        f"| Overall answer F1 | {payload['answer_f1']:.4f} |",
        f"| Answer type accuracy | {payload['answer_type_accuracy']:.4f} |",
        f"| Extractive answer F1 | {payload['by_answer_type']['extractive']['answer_f1']:.4f} |",
        f"| Free-form answer F1 | {payload['by_answer_type']['free_form']['answer_f1']:.4f} |",
        f"| Yes/no accuracy | {payload['yes_no_accuracy']:.4f} |",
        f"| Unanswerable accuracy | {payload['unanswerable_accuracy']:.4f} |",
        f"| Evidence F1 | {payload['evidence_f1']:.4f} |",
        "",
        "The companion JSON contains every prediction, gold answer type, evidence score, latency, and token usage.",
    ]
    return "\n".join(lines) + "\n"


def persist_qasper_evaluation(payload: dict[str, Any], output: Path) -> dict[str, str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = output.with_suffix(".md")
    markdown.write_text(render_qasper_markdown(payload), encoding="utf-8")
    return {"json": str(output), "markdown": str(markdown)}
