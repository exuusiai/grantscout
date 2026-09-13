# Development log

## 2026-09-13

### Retrieval and reporting

- Added Chinese-to-English query interpretation through the local model with a deterministic fallback and cache.
- Added relevance, recent, citation, and automatic ranking modes.
- Added three recommended arXiv papers with direct PDF links and print-to-PDF report export.
- Removed internal evidence tables, citation audit details, and run metadata from the user-facing report.
- Added model-based cross-paper finding synthesis instead of presenting extracted conclusion sentences as findings.

### Agent interaction

- Added multi-turn requirement clarification. Ambiguous terms are clarified before retrieval.
- Added automatic report execution when the requirement agent returns a resolved task.
- Retained explicit rerun support for regenerated reports after feedback.
- Changed the primary report flow so a resolved conversation immediately starts the streaming review; the report button is now explicitly a regeneration control.

### Knowledge base foundation

- Added project metadata and project-isolated corpus directories.
- Added asynchronous batch ingestion with queued, parsing, indexing, ready, and failed states.
- Added Markdown, PDF, PPTX, and plain-text parsing and a custom parser registry.
- Added project, document-status, upload, listing, and JSON export APIs.
- Added a drag-and-drop upload panel with live task polling.
- Added visible project creation, upload queue, ingestion completion, and failure feedback. Selecting or creating a project now switches the active source to that project's corpus.
- Added project existence validation before resolving a project corpus path.
- Added project and individual document deletion with synchronized removal from SQLite FTS.
- Added automatic report archival: generated Markdown reports are parsed and hot-indexed beside uploaded papers so later project searches can reuse them.
- Added project-scoped persistent conversations, message restoration, conversation switching, and deletion. The UI now follows a project-as-workspace and conversation-as-task structure.

### Validation and known boundaries

- Verified project creation and Markdown ingestion through the live server; the test document reached ready at 100% and was searchable in its project corpus.
- The current deployment is single-node and uses an in-process worker pool plus SQLite FTS. Jobs do not yet survive a process crash.
- Project isolation is a storage boundary, not authentication or authorization.
- Persistent chat memory, PII redaction, layout-aware table/formula recovery, reviewed comparability judgments, reproducibility scoring, collaboration audit history, and scheduled topic monitoring remain gated work recorded in `product-roadmap.md`.
