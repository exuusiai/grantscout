# PaperScout Architecture

PaperScout keeps the research workflow explicit and inspectable:

```text
Question
  -> Conversation and constraint clarification
  -> Source selection (arXiv or project knowledge base)
  -> Query interpretation and paper/document search
  -> Evidence Retrieval
  -> Structured Fact Extraction
  -> Five-axis Comparability Gate
  -> Cross-Paper Support / Conflict Analysis
  -> Research Decisions
  -> Claim / Citation Audit
  -> Localized HTML note + Markdown archive + JSON state
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
- Uploads run through an in-process worker pool. A successful task progresses through
  parsing and indexing to `ready`, after which it is immediately searchable through
  the project's SQLite FTS corpus. Uploading does not change the active search source;
  source changes are explicit user actions.

## Model routing

- The conversation model resolves ambiguity and extracts research constraints.
- An optional, separately configured OpenAI-compatible research model performs
  structured fact extraction and evidence-bounded synthesis.
- Both integrations use `/chat/completions`; PaperScout does not use `/responses`.
- Reports are rendered deterministically from the final research state. Chinese is the
  default locale, and the UI keeps separate Chinese and English report fragments.

## Deployment boundary

The current deployment is a single-node research prototype: project separation is a
filesystem and database boundary, the queue is process-local, and SQLite is the source
of truth. Authentication, durable distributed jobs, object-level authorization, and
multi-node storage are roadmap items rather than implemented guarantees.

## Runtime modes

The default mode is deterministic and offline. It is useful for tests, fixed
benchmarks, and environments without a model server. An OpenAI-compatible server
can be configured through `.env`; vLLM startup is provided in `scripts/start_vllm.sh`.

The semantic index is intentionally separate from the SQLite corpus. Rebuilding
the corpus never silently changes a vector index; run `paperscout index` after
ingesting new papers.
