# Failure Cases

These cases are deliberately retained as regression tests. They describe known
system boundaries rather than claims about model quality.

## 1. No Local Evidence

`test_no_match_retains_missing_evidence_warning` queries a corpus about grounded
retrieval with an unrelated coral-bleaching question. PaperScout returns no paper,
no claim, and an explicit missing-evidence warning. The system does not synthesize
an unsupported answer.

## 2. Claim Without Citation

`test_missing_citation_is_audit_failure` attaches no evidence ID to a claim. The
citation audit returns `failed` and names the missing reference. This prevents an
uncited sentence from being presented as a supported finding.

## 3. Unsupported Document Format

`test_unsupported_document_extension_fails_explicitly` supplies an HTML file. The
parser rejects it with an actionable `ValueError`; it does not treat arbitrary web
content as a paper or execute anything in it.

## 4. Opposing Outcome Language

`test_cross_paper_conflict_keeps_both_evidence_ids` supplies one positive and one
negative conclusion from different papers. The conflict record preserves both
evidence IDs so reviewers can inspect experimental conditions instead of receiving
a forced combined conclusion.

## 5. Invalid Evaluation Record

`test_evaluation_query_without_query_field_fails_explicitly` provides a JSONL
record without `query`. Evaluation fails before metrics are calculated, avoiding a
quietly malformed benchmark result.

Run all five cases with:

```bash
pytest -q tests/test_failure_cases.py
```
