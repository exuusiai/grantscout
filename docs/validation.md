# Human and NLI Validation

## Current engineering validation

As of 2026-09-14, the focused Web/report/knowledge regression suite passes 19 tests.
The live AutoDL corpus was also checked with previously uploaded PPTX documents: the
documents were `ready`, present in project-isolated SQLite corpora, and returned by an
FTS query. This validates the current upload-to-search path only.

The Chinese report regression checks localized claim status, comparability verdicts,
five comparison axes, reproduction recommendations, deterministic readiness reasons,
and missing-information labels. It also checks that the corresponding fixed English
pipeline phrases do not leak into the Chinese fragment. Paper-specific free-form text
can remain in its source language when no trustworthy localized field exists.

These checks do not establish production retrieval quality, PDF layout fidelity,
conflict precision/recall, or multi-user isolation. Those require the evaluations below.

GrantScout separates two tasks that require different labels:

1. **Claim-evidence entailment** asks whether a cited passage supports,
   contradicts, or does not establish one generated claim.
2. **Cross-paper conflict** asks whether two claims about the same subject,
   intervention, outcome, population, and experimental setting are mutually
   incompatible.

SciFact SUPPORT/CONTRADICT labels are suitable for the first task. They are not
ground truth for cross-paper conflict, because they relate one corpus abstract
to a benchmark claim rather than two GrantScout claims to each other.

## Qasper metric contract

Qasper answer artifacts include `metrics_version`. Version `2.0` separates
lexical overlap, answer type, and strict correctness:

- `answer_f1` is the maximum token F1 against any annotation.
- `answer_exact_match` ignores answer type and matches normalized text.
- `answer_type_accuracy` matches the predicted type against any annotated type.
- `type_and_answer_accuracy` requires type and text to match the same annotation.
- `yes_no_accuracy` and `unanswerable_accuracy` use that strict rule on their subsets.

One question may belong to multiple type subsets because Qasper has multiple
annotators. These subsets are diagnostic views and must not be summed. Version
1 and version 2 class accuracies are only comparable after rescoring the same
predictions.

On the 160-question stratified set, increasing generation context from K=5 to
K=10 raised Evidence F1 from `0.2865` to `0.3316`, but reduced Answer F1 from
`0.3705` to `0.3536` and increased mean generation latency from `1.1963s` to
`1.5656s`. The next experiment should retrieve 10 or 20 candidates, rerank or
compress them to five passages for generation, and compare both fixed-context
baselines.

## Recommended validation design

- Draw a stratified sample of at least 300 claim-evidence pairs: supported,
  unsupported, and suspected contradiction, including low- and high-confidence
  cases. Keep a separate untouched test split.
- Use two independent domain-capable annotators. Show the exact claim, cited
  passage, paper title, and enough local context; hide the system prediction.
- Label `entailed`, `contradicted`, `insufficient`, or `invalid/unclear`. Record
  a short rationale and the minimal supporting span. Adjudicate disagreements
  with a third reviewer and report Cohen's kappa or Krippendorff's alpha.
- For cross-paper conflict, additionally label whether subject, intervention,
  outcome, population, and setting align. Only call the pair a conflict when
  those fields align and the conclusions are incompatible.
- Freeze the annotation guide after a 30-50 item pilot. Never tune thresholds on
  the final test split.

## NLI model use

Use a locally hosted scientific-domain NLI or claim-verification model as a
screening and scoring layer, not as the sole ground truth. Evaluate it against
the adjudicated set and calibrate entailment/contradiction thresholds on the
development split. Report per-class precision, recall, F1, a confusion matrix,
coverage at the chosen abstention threshold, and bootstrap confidence
intervals. Route low-confidence and premise-truncated cases to human review.

A robust production cascade is:

`retrieval -> deterministic citation checks -> local NLI -> confidence-based abstention -> human review`

Store model name, revision, prompt/template, premise truncation, thresholds,
and dataset hashes with every evaluation artifact. This makes the result
auditable and prevents an NLI score from being presented as human factuality.
