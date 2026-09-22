# Evaluation

Evaluation query files are JSONL. Each line has a query and one or both relevance
lists:

```json
{"query":"evidence recall","relevant_paper_ids":["paper-1"],"relevant_evidence_ids":["paper-1:section:0001:evidence:0000"]}
```

Commands:

```bash
grantscout evaluate evals/custom_topics/queries.jsonl --corpus data/corpus.sqlite
grantscout benchmark evals/custom_topics/queries.jsonl --corpus data/corpus.sqlite
grantscout ablate evals/custom_topics/queries.jsonl --corpus data/corpus.sqlite
grantscout evaluate-suite evals/custom_topics/queries.jsonl --corpus data/corpus.sqlite
```

`evaluate-suite` writes a timestamped JSON source artifact and a Markdown
comparison report under `runs/`. It includes per-query evidence IDs and Agent run
IDs, a deterministic single-pass context baseline, fixed RAG, Agentic RAG, and five
variants: full, without question decomposition, without reranking, without citation
audit, and without conflict detection.

## Official Dataset Adapters

SciFact claims are converted into query records with
`scripts/build_scifact_eval.py`. The script preserves the official cited paper
IDs and maps annotated abstract sentence indices to stable GrantScout evidence
IDs. The resulting file can be evaluated with either retrieval mode:

```bash
python scripts/build_scifact_eval.py \
  --claims data/raw/scifact/claims_dev.jsonl \
  --corpus-jsonl data/raw/scifact/corpus.jsonl \
  --corpus data/scifact.sqlite \
  --output evals/scifact/dev.jsonl
grantscout evaluate evals/scifact/dev.jsonl \
  --corpus data/scifact.sqlite --top-k 5 --mode lexical
grantscout evaluate evals/scifact/dev.jsonl \
  --corpus data/scifact.sqlite --top-k 5 --mode semantic
```

Qasper is imported as a paper corpus and its question/evidence annotations are
converted with `scripts/build_qasper_eval.py`:

```bash
grantscout ingest-jsonl data/raw/qasper/dev.json --corpus data/qasper.sqlite
python scripts/build_qasper_eval.py \
  --source data/raw/qasper/dev.json \
  --corpus data/qasper.sqlite \
  --output evals/qasper/dev.jsonl
grantscout evaluate evals/qasper/dev.jsonl \
  --corpus data/qasper.sqlite --top-k 5 --mode semantic
```

Qasper answer quality is not inferred from retrieval scores. The generated
records retain the official answer payload as `qasper_answer`; answer
correctness still requires a task-specific QA evaluator or human labels.

GrantScout provides the task-specific evaluator through
`evaluate-qasper-answers`. It retrieves evidence, asks only the configured local
chat-completions endpoint for a typed answer, rejects invented evidence IDs, and
scores against all Qasper annotators:

```bash
GRANTSCOUT_VECTOR_INDEX_PATH=data/qasper.bge-small-en-v1.5.auto.index \
grantscout evaluate-qasper-answers evals/qasper/dev.jsonl \
  --corpus data/qasper.sqlite --mode semantic --top-k 5 --workers 4 \
  --checkpoint runs/qasper-answer-predictions.jsonl \
  --output runs/qasper-answer-evaluation.json
```

The output includes overall token F1, answer-type accuracy, extractive and
free-form F1, yes/no accuracy, unanswerable accuracy, evidence precision/recall/F1,
per-query latency, aggregate token usage, and model failures. Predictions are
checkpointed after every completed request, so rerunning the same command resumes
instead of discarding finished work. JSON retains every prediction and score;
the companion Markdown is a concise summary.

## SciFact Dev Results

The formal dev run contains 300 claims, of which 188 have official evidence
annotations. At `K=5`, the local semantic index materially outperforms the
dependency-free lexical baseline:

| Mode | Recall@5 | Evidence Recall@5 | MRR | Evidence-labeled queries |
| --- | ---: | ---: | ---: | ---: |
| Lexical | 0.0517 | 0.0532 | 0.0227 | 188 |
| Semantic (BGE) | 0.7479 | 0.8575 | 0.6606 | 188 |

These metrics measure retrieval only. They do not claim that a generated report
is factually correct, that a cited passage entails the whole claim, or that
Qasper answer quality has been evaluated. The raw artifacts are
`runs/scifact-dev-lexical-evaluation-v2.json` and
`runs/scifact-dev-semantic-evaluation.json`; the concise comparison is in
`docs/scifact-retrieval-comparison.md`.

## Qasper Dev Results

The Qasper dev adapter produced 1,005 question records, including 923 with
non-empty evidence annotations. At `K=5`, the local semantic index improves
paper-level retrieval over lexical retrieval, while evidence recall remains a
separate limitation of this first adapter:

| Mode | Recall@5 | Evidence Recall@5 | MRR | Evidence-labeled queries |
| --- | ---: | ---: | ---: | ---: |
| Lexical | 0.0458 | 0.0019 | 0.0281 | 923 |
| Semantic (BGE) | 0.3343 | 0.1066 | 0.2776 | 923 |

The raw artifacts are `runs/qasper-dev-lexical-evaluation.json` and
`runs/qasper-dev-semantic-evaluation.json`. The semantic result uses a Qasper
specific index (`data/qasper.bge-small-en-v1.5.auto.index`), rather than the
SciFact index used by the default Web UI. Qasper answer correctness is not
included in these retrieval metrics.

The source-paper-constrained Qasper answer evaluation completed all 1,005 dev
questions with Qwen3-8B and no model failures. It finished in 415.714 seconds
with mean generation latency of 1.5914 seconds:

| Metric | Score |
| --- | ---: |
| Overall answer F1 | 0.2519 |
| Answer type accuracy | 0.5453 |
| Extractive answer F1 | 0.3222 |
| Free-form answer F1 | 0.2560 |
| Yes/no accuracy | 0.0775 |
| Unanswerable accuracy | 0.3259 |
| Evidence F1 | 0.3238 |

The evaluator restricts retrieval to the paper associated with each Qasper
question. This follows the dataset task definition and prevents evidence from
unrelated papers from contaminating answer scores.

## SciFact Offline Evaluation Suite

The complete offline suite was run with semantic retrieval and model reasoning
disabled, so its results are deterministic retrieval/agent-flow measurements.
The artifact is `runs/20260909T174228Z-evaluation-20cb7615.json` with the
human-readable report at `runs/20260909T174228Z-evaluation-20cb7615.md`.

| System or variant | Paper Recall | Evidence Recall |
| --- | ---: | ---: |
| Single-pass context | 0.0517 | 0.0333 |
| Fixed RAG | 0.0517 | 0.0333 |
| GrantScout Agentic RAG | 0.7452 | 0.5501 |
| Ablation: without question decomposition | 0.7611 | 0.5693 |
| Ablation: without reranker | 0.7452 | 0.5501 |
| Ablation: without citation audit | 0.7452 | 0.5501 |
| Ablation: without conflict detection | 0.7452 | 0.5501 |

The suite duration was 633.116 seconds. The benchmark table intentionally
keeps the fixed baselines lexical, while GrantScout Agentic RAG follows the
configured semantic retrieval mode. Citation audit and conflict detection
affect traceability and warnings; they are not expected to change retrieval
recall in this implementation.

New suite artifacts report metrics aligned to those modules:

- `citation_coverage`: fraction of claims carrying at least one evidence ID.
- `audit_execution_rate`: fraction of runs where citation audit executed.
- `citation_support_precision`: audited claims passing the evidence support check.
- `unsupported_claim_rate`: audited claims failing that check.
- `conflict_precision`, `conflict_recall`, `conflict_f1`: computed only when the
  query record contains an independently adjudicated `conflict_expected` label.
- Mean model calls and tool duration for execution-cost comparisons.

`citation_support_precision` is an internal groundedness metric, not a substitute
for human factuality labels. SciFact conversion deliberately does not derive
cross-paper conflict labels from its `CONTRADICT` evidence labels: those labels
describe claim-evidence relations. Disabling citation audit produces `N/A`
support precision rather than a misleading unchanged score. Use
`scripts/evaluate_conflict_annotations.py` with independently reviewed pairs for
cross-paper conflict evaluation.

## Retrieval Performance

The lexical backend uses SQLite FTS5/BM25 and automatically backfills its index
when an existing corpus is first opened. It updates FTS rows transactionally on
paper upsert and falls back to the original scan implementation if SQLite was
built without FTS5. Baseline comparison reuses one lexical result set per query,
and a semantic suite loads one BGE model/index instance for retrieval, benchmark,
and all ablation variants.

These are retrieval metrics; claim entailment and human report quality require
labeled annotations. The single-pass baseline is an offline retrieval-context proxy;
when a local vLLM endpoint is available, it can be extended with generated-answer
quality labels without changing the stored retrieval evidence.

The post-FTS SciFact dev run completed in 7.726 seconds. At `K=5`, FTS5/BM25
achieved Recall=`0.6967`, evidence recall=`0.7685`, and MRR=`0.6043`. The
optimized semantic suite completed in 101.844 seconds, compared with 633.116
seconds before semantic-index and baseline-result reuse.

The earlier `64 positive / 236 negative` conflict numbers were generated by an
invalid conversion of SciFact `CONTRADICT` labels and are retired. They must not
be cited as cross-paper conflict performance. The current 30-pair silver pilot
has no conflict positives, so its audit reports observed false positives only and
leaves precision/recall undefined. Human conflict labels are required before a
formal conflict score can be reported.
