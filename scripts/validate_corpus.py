#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from grantscout.retrieval.store import CorpusStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with CorpusStore(args.corpus) as store:
        papers = store.list_papers()
        evidence = store.list_evidence()
        result = {
            "papers": len(papers),
            "evidence_items": len(evidence),
            "papers_without_evidence": [paper.id for paper in papers if not any(item.paper_id == paper.id for _, item in evidence)],
        }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
