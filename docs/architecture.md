# PaperScout Architecture

PaperScout keeps the research workflow explicit and inspectable:

```text
Question
  -> Question Planner
  -> Paper Search
  -> Evidence Retrieval
  -> Structured Fact Extraction
  -> Cross-Paper Comparison
  -> Claim / Citation Audit
  -> Markdown + JSON + JSONL trajectory
```

## Storage

- `papers`, `sections`, and `evidence` live in SQLite.
- Evidence IDs are stable: `paper_id:section:NNNN:evidence:NNNN`.
- Every run writes a state JSON file, a JSONL tool trajectory, and a Markdown report.
- The lexical index is dependency-free. The optional semantic index uses BGE-M3 and
  FAISS, with a NumPy fallback when FAISS is unavailable.
- Project knowledge bases contain only user-uploaded/collected documents and generated
  reports. External arXiv/OpenAlex searches use query-scoped caches of public metadata
  and abstracts; PaperScout does not download remote PDFs into project storage.

## Runtime modes

The default mode is deterministic and offline. It is useful for tests, fixed
benchmarks, and environments without a model server. An OpenAI-compatible server
can be configured through `.env`; vLLM startup is provided in `scripts/start_vllm.sh`.

The semantic index is intentionally separate from the SQLite corpus. Rebuilding
the corpus never silently changes a vector index; run `paperscout index` after
ingesting new papers.
