#!/usr/bin/env python3
"""Fill paper metadata (venue, cited_by_count) from the OpenAlex index.

Network is required only when this script runs; the product itself stays
offline-capable. Papers are matched by title search, and results are
written into the corpus metadata columns used by @{impact}/@{conference}.

Usage:
    grantscout-python scripts/enrich_metadata.py --corpus data/corpus.sqlite
"""

import json
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grantscout.retrieval.store import CorpusStore  # noqa: E402

OPENALEX_URL = "https://api.openalex.org/works"


def lookup_title(client: httpx.Client, title: str) -> dict | None:
    clean = re.sub(r"[^\w\s-]", "", title).strip()[:120]
    for attempt in range(4):
        try:
            response = client.get(
                OPENALEX_URL,
                params={"filter": f"title.search:{clean}", "per-page": 1, "mailto": "grantscout@localhost"},
                timeout=15.0,
            )
            if response.status_code == 429:
                wait = 10.0 * (attempt + 1)
                print(f"  429 rate limited, waiting {wait:.0f}s ...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            results = response.json().get("results", [])
            return results[0] if results else None
        except httpx.HTTPError as error:
            if attempt == 3:
                raise
            print(f"  retry after error: {error}")
            time.sleep(5.0 * (attempt + 1))
    return None


def main() -> int:
    if len(sys.argv) < 3 or sys.argv[1] != "--corpus":
        print(__doc__)
        return 2
    corpus_path = Path(sys.argv[2])
    if not corpus_path.exists():
        print(f"Corpus does not exist: {corpus_path}")
        return 1
    updated, missed = 0, 0
    with httpx.Client() as client, CorpusStore(corpus_path) as store:
        for paper in store.list_papers():
            try:
                work = lookup_title(client, paper.title)
            except httpx.HTTPError as error:
                print(f"lookup failed for {paper.id}: {error}")
                missed += 1
                time.sleep(3.0)
                continue
            if work is None:
                missed += 1
                continue
            venue = None
            primary = work.get("primary_location") or {}
            source = primary.get("source") or {}
            venue = source.get("display_name")
            store.set_paper_meta(
                paper.id,
                venue=venue,
                cited_by_count=int(work.get("cited_by_count") or 0),
            )
            updated += 1
            time.sleep(2.0)
    print(json.dumps({"updated": updated, "missed": missed}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
