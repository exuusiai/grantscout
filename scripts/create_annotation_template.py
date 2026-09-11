#!/usr/bin/env python3
"""Create a blinded human/NLI validation annotation JSONL template."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


VALID_LABELS = {"entailed", "contradicted", "insufficient", "invalid/unclear"}


def build_template(source: Path, output: Path, limit: int | None = None) -> int:
    rows: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        claim = str(item.get("claim") or item.get("query") or "")
        evidence = item.get("evidence") or item.get("evidence_text") or ""
        if isinstance(evidence, list):
            evidence = "\n\n".join(str(value) for value in evidence)
        key = f"{item.get('id', '')}\n{claim}\n{evidence}".encode()
        rows.append(
            {
                "annotation_id": hashlib.sha256(key).hexdigest()[:16],
                "source_id": item.get("id"),
                "claim": claim,
                "evidence": str(evidence),
                "annotator_a": {"label": None, "rationale": "", "minimal_span": ""},
                "annotator_b": {"label": None, "rationale": "", "minimal_span": ""},
                "adjudication": {"label": None, "rationale": "", "reviewer": ""},
                "labels": ["entailed", "contradicted", "insufficient", "invalid/unclear"],
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return len(rows)


def validate_annotations(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed = agreements = 0
    invalid: list[str] = []
    for row in rows:
        first = row.get("annotator_a", {}).get("label")
        second = row.get("annotator_b", {}).get("label")
        if first not in VALID_LABELS or second not in VALID_LABELS:
            invalid.append(str(row.get("annotation_id")))
            continue
        completed += 1
        agreements += first == second
    return {
        "records": len(rows),
        "completed_pairs": completed,
        "raw_agreement": round(agreements / completed, 4) if completed else None,
        "invalid_or_incomplete_ids": invalid,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--validate", type=Path, default=None)
    args = parser.parse_args()
    if args.validate is not None:
        print(json.dumps(validate_annotations(args.validate), ensure_ascii=False))
        return
    if args.source is None or args.output is None:
        parser.error("--source and --output are required unless --validate is used")
    count = build_template(args.source, args.output, args.limit)
    print(json.dumps({"output": str(args.output), "records": count}))


if __name__ == "__main__":
    main()
