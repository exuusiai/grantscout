# SciFact Dev Retrieval Comparison

This report compares PaperScout retrieval on the official SciFact dev claims
split. The split contains 300 queries; 188 queries have official evidence
annotations. Results are reported at `K=5`.

| Mode | Recall@5 | Evidence Recall@5 | MRR | Evidence-labeled queries |
| --- | ---: | ---: | ---: | ---: |
| Lexical | 0.0517 | 0.0532 | 0.0227 | 188 |
| Semantic (BGE) | 0.7479 | 0.8575 | 0.6606 | 188 |

Semantic retrieval uses the local BGE embedding model and a local FAISS/NumPy
index. Neither run sends data to a third-party model gateway. Evidence Recall
is averaged only over queries with non-empty official evidence annotations;
unlabeled queries remain in paper Recall and MRR.

Raw JSON artifacts:

- `runs/scifact-dev-lexical-evaluation-v2.json`
- `runs/scifact-dev-semantic-evaluation.json`

Interpretation: semantic retrieval improves both paper-level and evidence-level
retrieval substantially on this split. These are retrieval metrics only. They
do not replace claim entailment checks, answer-quality evaluation, or human
review of the final literature report.
