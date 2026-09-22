#!/usr/bin/env python3
"""Build GrantScout evaluation JSONL from a Qasper split and local corpus."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterator

from grantscout.evaluation.datasets import iter_records
from grantscout.retrieval.store import CorpusStore


def _paper_id(record: dict[str, Any], index: int) -> str:
    return str(record.get("id") or record.get("paper_id") or f"paper-{index:06d}")


def _at(value: Any, index: int) -> Any:
    if isinstance(value, list) and index < len(value):
        return value[index]
    return value


def _qas(record: dict[str, Any]) -> Iterator[tuple[int, dict[str, Any]]]:
    qas = record.get("qas") or []
    if isinstance(qas, list):
        for index, item in enumerate(qas):
            if isinstance(item, dict):
                yield index, item
        return
    if not isinstance(qas, dict):
        return
    questions = qas.get("question") or []
    if isinstance(questions, str):
        questions = [questions]
    for index, question in enumerate(questions):
        yield index, {
            "question": question,
            "question_id": _at(qas.get("question_id"), index),
            "evidence": _at(qas.get("evidence"), index),
            "answer": _at(qas.get("answer"), index),
        }


def _locations(value: Any) -> Iterator[tuple[str, int]]:
    """Yield Qasper [section_name, paragraph_index] pairs at any nesting depth."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        section, paragraph = value
        if isinstance(section, str) and isinstance(paragraph, int):
            yield section, paragraph
            return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _locations(item)


def _answer_evidence_texts(qa: dict[str, Any]) -> Iterator[str]:
    """Yield Qasper's highlighted paragraph snippets and legacy evidence text."""
    answers = qa.get("answers") or []
    if isinstance(answers, dict):
        answers = [answers]
    for annotation in answers:
        if not isinstance(annotation, dict):
            continue
        answer = annotation.get("answer", annotation)
        if not isinstance(answer, dict):
            continue
        snippets = answer.get("highlighted_evidence") or answer.get("evidence") or []
        if isinstance(snippets, str):
            snippets = [snippets]
        for snippet in snippets:
            if str(snippet).strip():
                yield str(snippet)


def _text_terms(value: str) -> set[str]:
    return {term for term in re.findall(r"[\w]+", value.lower()) if len(term) > 2}


def _evidence_ids_for_qa(
    qa: dict[str, Any], sections: dict[str, Any], by_section: dict[str, list[Any]]
) -> list[str]:
    output: list[str] = []
    for section_title, paragraph_index in _locations(qa.get("evidence")):
        section = sections.get(section_title)
        evidence_items = by_section.get(section.id, []) if section else []
        if 0 <= paragraph_index < len(evidence_items):
            evidence_id = evidence_items[paragraph_index].id
            if evidence_id not in output:
                output.append(evidence_id)
    evidence_items = [item for items in by_section.values() for item in items]
    for snippet in _answer_evidence_texts(qa):
        target_terms = _text_terms(snippet)
        if not target_terms:
            continue
        ranked = sorted(
            evidence_items,
            key=lambda item: len(target_terms & _text_terms(item.text)) / len(target_terms),
            reverse=True,
        )
        if ranked:
            score = len(target_terms & _text_terms(ranked[0].text)) / len(target_terms)
            if score >= 0.2 and ranked[0].id not in output:
                output.append(ranked[0].id)
    return output


def build_records(source: Path, corpus: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records = list(iter_records(source))
    if limit is not None:
        records = records[:limit]
    output: list[dict[str, Any]] = []
    with CorpusStore(corpus) as store:
        evidence_by_paper: dict[str, dict[str, list[Any]]] = {}
        for paper, evidence in store.list_evidence():
            evidence_by_paper.setdefault(paper.id, {}).setdefault(evidence.section_id, []).append(evidence)
        for index, record in enumerate(records):
            paper_id = _paper_id(record, index)
            sections = {section.title: section for section in store.get_sections(paper_id)}
            by_section = evidence_by_paper.get(paper_id, {})
            for qa_index, qa in _qas(record):
                question = str(qa.get("question") or "").strip()
                if not question:
                    continue
                evidence_ids = _evidence_ids_for_qa(qa, sections, by_section)
                output.append(
                    {
                        "id": str(qa.get("question_id") or f"{paper_id}:q{qa_index:04d}"),
                        "query": question,
                        "relevant_paper_ids": [paper_id],
                        "relevant_evidence_ids": evidence_ids,
                        "qasper_answer": qa.get("answers") or qa.get("answer"),
                    }
                )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Qasper train/dev/test JSON file")
    parser.add_argument("--corpus", type=Path, required=True, help="Ingested SQLite corpus")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    records = build_records(args.source, args.corpus, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(records),
                "with_evidence_labels": sum(bool(item["relevant_evidence_ids"]) for item in records),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
