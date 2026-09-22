#!/usr/bin/env bash
set -euo pipefail

CORPUS="${1:-data/demo.sqlite}"
python -m grantscout ingest-jsonl examples/demo_corpus.jsonl --corpus "$CORPUS"
python -m grantscout evaluate examples/demo_queries.jsonl --corpus "$CORPUS" --top-k 5
python -m grantscout benchmark examples/demo_queries.jsonl --corpus "$CORPUS" --top-k 5
python -m grantscout ablate examples/demo_queries.jsonl --corpus "$CORPUS" --top-k 5
python -m grantscout evaluate-suite examples/demo_queries.jsonl --corpus "$CORPUS" --top-k 5
python -m grantscout ask "Which methods improve evidence recall and what are their limitations?" --corpus "$CORPUS"
