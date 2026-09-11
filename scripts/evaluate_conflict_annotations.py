#!/usr/bin/env python3
"""Audit cross-paper conflict annotations without overstating silver-label evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


LABELS = {"conflict", "compatible", "not_comparable", "insufficient", "invalid/unclear"}


def _paper_pair_from_evidence(conflict: dict[str, Any]) -> frozenset[str]:
    ids = list(conflict.get("positive_evidence_ids", [])) + list(
        conflict.get("negative_evidence_ids", [])
    )
    return frozenset(str(value).split(":section:", 1)[0] for value in ids)


def evaluate(annotations: Path, source_runs: list[Path]) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in annotations.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    predicted: set[tuple[str, frozenset[str]]] = set()
    for path in source_runs:
        run = json.loads(path.read_text(encoding="utf-8"))
        run_id = str(run.get("run_id") or "")
        predicted.update(
            (run_id, pair)
            for item in run.get("conflicts", [])
            if len(pair := _paper_pair_from_evidence(item)) == 2
        )

    model_counts: Counter[str] = Counter()
    human_counts: Counter[str] = Counter()
    invalid_ids: list[str] = []
    false_positives = true_positives = false_negatives = 0
    evaluated = 0
    for row in rows:
        model_label = row.get("model_annotation", {}).get("label")
        if model_label in LABELS:
            model_counts[str(model_label)] += 1
        adjudicated = row.get("adjudication", {}).get("label")
        first = row.get("annotator_a", {}).get("label")
        second = row.get("annotator_b", {}).get("label")
        human_label = adjudicated or (first if first == second else None)
        if human_label is not None and human_label not in LABELS:
            invalid_ids.append(str(row.get("annotation_id")))
            continue
        if human_label is not None:
            human_counts[str(human_label)] += 1
        label = human_label or model_label
        if label not in LABELS:
            continue
        paper_pair = frozenset(str(item["paper_id"]) for item in row.get("papers", []))
        system_positive = (str(row.get("source_run_id") or ""), paper_pair) in predicted
        gold_positive = label == "conflict"
        evaluated += 1
        true_positives += system_positive and gold_positive
        false_positives += system_positive and not gold_positive
        false_negatives += not system_positive and gold_positive

    positives = true_positives + false_negatives
    precision_denominator = true_positives + false_positives
    return {
        "annotation_records": len(rows),
        "label_source": "human" if sum(human_counts.values()) == len(rows) else "silver_or_mixed",
        "model_label_distribution": dict(sorted(model_counts.items())),
        "human_label_distribution": dict(sorted(human_counts.items())),
        "evaluated_pairs": evaluated,
        "system_predicted_conflicts": len(predicted),
        "false_positives": false_positives,
        "observed_false_positive_rate": round(false_positives / evaluated, 4) if evaluated else None,
        "precision": round(true_positives / precision_denominator, 4) if precision_denominator else None,
        "recall": round(true_positives / positives, 4) if positives else None,
        "recall_identifiable": positives > 0,
        "invalid_annotation_ids": invalid_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = evaluate(args.annotations, args.source_run)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
