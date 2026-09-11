#!/usr/bin/env python3
"""Build PaperScout evaluation JSONL from a local SciFact claims split."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from paperscout.retrieval.store import CorpusStore


def _terms(text: str) -> set[str]:
    return {term for term in re.findall(r"[\w]+", text.lower()) if len(term) > 2}


def _load_corpus_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[str(record["doc_id"])] = record
    return records


def _best_evidence_id(
    paper_id: str,
    sentence_text: str,
    evidence_by_paper: dict[str, list[Any]],
) -> str | None:
    candidates = evidence_by_paper.get(paper_id, [])
    if not candidates:
        return None
    target_terms = _terms(sentence_text)
    ranked = sorted(
        candidates,
        key=lambda item: len(target_terms & _terms(item.text)) / max(len(target_terms), 1),
        reverse=True,
    )
    return ranked[0].id if ranked and target_terms else candidates[0].id


def build_records(
    claims_path: Path,
    corpus_jsonl: Path,
    corpus_sqlite: Path,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    corpus_records = _load_corpus_records(corpus_jsonl)
    claims = [json.loads(line) for line in claims_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        claims = claims[:limit]
    with CorpusStore(corpus_sqlite) as store:
        evidence_by_paper: dict[str, list[Any]] = defaultdict(list)
        for paper, evidence in store.list_evidence():
            evidence_by_paper[paper.id].append(evidence)

    output: list[dict[str, Any]] = []
    for claim in claims:
        paper_ids = [str(value) for value in claim.get("cited_doc_ids", [])]
        evidence_ids: list[str] = []
        labels = claim.get("evidence") or {}
        for paper_id, annotations in labels.items():
            abstract = corpus_records.get(str(paper_id), {}).get("abstract") or []
            for annotation in annotations or []:
                for sentence_index in annotation.get("sentences", []):
                    if isinstance(abstract, list) and 0 <= sentence_index < len(abstract):
                        evidence_id = _best_evidence_id(
                            str(paper_id), str(abstract[sentence_index]), evidence_by_paper
                        )
                        if evidence_id and evidence_id not in evidence_ids:
                            evidence_ids.append(evidence_id)
        output.append(
            {
                "id": claim.get("id"),
                "query": str(claim["claim"]),
                "relevant_paper_ids": paper_ids,
                "relevant_evidence_ids": evidence_ids,
                "scifact_evidence_labels": labels,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--corpus-jsonl", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    records = build_records(args.claims, args.corpus_jsonl, args.corpus, args.limit)
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
                "with_evidence_labels": sum(bool(record["relevant_evidence_ids"]) for record in records),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
