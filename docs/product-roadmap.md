# Product roadmap

PaperScout is moving from a one-shot paper summarizer to a persistent research workspace.

## Implemented foundation

- Project-isolated corpora and document metadata.
- Batch asynchronous ingestion with observable queued, parsing, indexing, ready, and failed states.
- Markdown, PDF, PPTX, and text parsers with a custom parser registry.
- Hot indexing into SQLite FTS and project export.
- Project/document deletion and automatically indexed report archives.
- Project-scoped persistent conversations with restore, switch, and delete controls.
- Evidence-gated five-axis comparability and conservative conflict suppression.
- Structured research constraints plus evidence-bounded reproduction-readiness decisions.

SQLite remains the transactional source of truth for the current single-node deployment. Storage,
queue, and search interfaces should be measured before introducing PostgreSQL, Milvus, Typesense,
or Airbyte. A production multi-node deployment can replace these adapters independently.

## Next delivery gates

1. Authentication, memberships, object-level authorization, quotas, and encrypted object storage.
2. Presidio-based PII detection/redaction with audit records and per-project policy.
3. Reviewed long-term memory extraction and user-editable project instructions beyond persisted conversation history.
4. Layout-aware PDF/TeX parsing, tables, figures, formulas, citations, and provenance coordinates.
5. Human-calibrated thresholds for five-axis comparability and contradiction classification.
6. GitHub/code metadata ingestion to replace unknown fields in reproducibility readiness scoring.
7. Human edits, evidence labels, version history, and reviewer attribution.
8. Saved-topic incremental monitoring, report diffs, and notifications.

Each gate requires evaluation data and acceptance thresholds; these capabilities must not be
presented as reliable merely because an LLM can emit the corresponding JSON fields.
