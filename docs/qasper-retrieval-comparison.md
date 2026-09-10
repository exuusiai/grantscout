# Qasper Dev Retrieval Comparison

The Qasper dev adapter generated 1,005 question records from 281 papers. Of
these, 923 records contain non-empty official evidence annotations. Results are
reported at `K=5`.

| Mode | Recall@5 | Evidence Recall@5 | MRR | Evidence-labeled queries |
| --- | ---: | ---: | ---: | ---: |
| Lexical | 0.0458 | 0.0019 | 0.0281 | 923 |
| Semantic (BGE) | 0.3343 | 0.1066 | 0.2776 | 923 |

The semantic run uses the Qasper-specific local index
`data/qasper.bge-small-en-v1.5.auto.index`. The default Web UI remains pointed
at the SciFact corpus and SciFact index; switching corpora and index paths must
be explicit so that evidence IDs and vector rows cannot be mixed.

Raw artifacts:

- `runs/qasper-dev-lexical-evaluation.json`
- `runs/qasper-dev-semantic-evaluation.json`

These are retrieval metrics. The generated `qasper_answer` payload is preserved
in `evals/qasper/dev.jsonl`, but answer correctness still requires a Qasper QA
evaluator or human review.
