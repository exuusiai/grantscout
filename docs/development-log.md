# Development log

## 2026-09-14

### Reliable comparison and decisions

- Added pairwise five-axis comparability assessments for task, dataset/split, metric, scale/budget, and experimental conditions.
- Changed contradiction detection to emit conflicts only for paper pairs that pass the comparability gate; missing evidence is reported as insufficient rather than treated as equality.
- Added structured user constraints for time range, open-source requirement, model size, VRAM, dataset preference, code requirement, and quality/speed/cost priority.
- Added conservative reproduction-readiness scores and research decisions. Unknown code, license, dependency, and hardware facts are penalized and listed instead of inferred.
- Added comparability and research-decision sections to the bilingual report.

### Workspace and external search boundaries

- Reorganized the Web UI into a Codex-style three-pane workspace: project/conversation/private knowledge sidebar, primary chat surface, and collapsible run evidence plus report panel.
- Added a user-controlled paper investigation limit from 1 to 20 (default 5), with conversation extraction for requests such as "analyze 12 papers".
- Removed the former hard-coded three-paper arXiv analysis limit.
- Clarified the storage boundary: external arXiv/OpenAlex search stores only temporary query metadata and abstracts; it never downloads paper PDFs into the private project knowledge base.
- Added explicit collection of a recommended paper into the current project. Collection stores searchable title, authors, abstract, and source URL only after user action; it does not download the remote PDF.
- Reworked the sidebar to match a conventional Codex-style model application: persistent brand and new-chat action, expandable project folders with nested conversations, and secondary knowledge/project administration anchored at the bottom. The global duplicate header was removed and the report gained an independent visibility toggle.

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

### 2026-09-14 reliability and model routing

- Removed whole-sidebar polling behavior; the project tree is refreshed only by explicit project/conversation events.
- Added a compatibility layer so review requests always carry project, locale, ranking, paper count, and research constraints.
- Added optional `PAPERSCOUT_RESEARCH_MODEL_*` settings for a dedicated local OpenAI-compatible research model. Conversation understanding continues to use the built-in model; paper analysis uses the research profile when configured, with local fallback.
- Corrected arXiv result sizing to honor the requested limit and configured maximum.
- Empty project knowledge bases now fall back to arXiv with a visible warning instead of failing on a missing `corpus.sqlite` file. Selecting a project can therefore remain independent from the external paper search scope.
- Specialized paper analysis now extracts comparison-critical conditions, synthesizes applicability boundaries, and reports three-state comparability (`comparable`, `condition_mismatch`, `insufficient_evidence`) with five-axis values and evidence references. Pairwise comparison is bounded to the eight most evidence-rich papers to keep 20-paper reviews useful and tractable.
- Project isolation is a storage boundary, not authentication or authorization.
- Persistent chat memory, PII redaction, layout-aware table/formula recovery, reviewed comparability judgments, reproducibility scoring, collaboration audit history, and scheduled topic monitoring remain gated work recorded in `product-roadmap.md`.
