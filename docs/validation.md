# Human and NLI Validation

PaperScout separates two tasks that require different labels:

1. **Claim-evidence entailment** asks whether a cited passage supports,
   contradicts, or does not establish one generated claim.
2. **Cross-paper conflict** asks whether two claims about the same subject,
   intervention, outcome, population, and experimental setting are mutually
   incompatible.

SciFact SUPPORT/CONTRADICT labels are suitable for the first task. They are not
ground truth for cross-paper conflict, because they relate one corpus abstract
to a benchmark claim rather than two PaperScout claims to each other.

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
