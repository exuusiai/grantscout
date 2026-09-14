from itertools import combinations

from paperscout.models.schemas import ComparabilityAssessment, Conflict, StructuredFacts


def compare_papers(
    facts: list[StructuredFacts], prefer_localized: bool = False
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in facts:
        rows.append(
            {
                "paper_id": item.paper_id,
                "methods": [_display(fact, prefer_localized) for fact in item.methods[:5]],
                "datasets": [_display(fact, prefer_localized) for fact in item.datasets[:5]],
                "experimental_settings": [_display(fact, prefer_localized) for fact in item.experimental_settings[:5]],
                "metrics": [_display(fact, prefer_localized) for fact in item.metrics[:5]],
                "conclusions": [_display(fact, prefer_localized) for fact in item.conclusions[:5]],
                "limitations": [_display(fact, prefer_localized) for fact in item.limitations[:5]],
            }
        )
    return rows


def _display(fact, prefer_localized: bool) -> str:
    return fact.localized_text if prefer_localized and fact.localized_text else fact.text


def assess_comparability(facts: list[StructuredFacts]) -> list[ComparabilityAssessment]:
    assessments = []
    for left, right in combinations(facts, 2):
        datasets = _overlap(left.datasets, right.datasets)
        metrics = _overlap(left.metrics, right.metrics)
        settings = _overlap(left.experimental_settings, right.experimental_settings)
        task = _overlap(left.methods, right.methods) or _overlap(left.conclusions, right.conclusions)
        known = [task, datasets, metrics, settings]
        comparable = all(value is True for value in (task, datasets, metrics)) and settings is not False
        if comparable:
            reason = "Same task, dataset, and metric are evidenced; experimental conditions do not conflict."
        elif any(value is False for value in known):
            reason = "Key task, dataset, metric, or experimental conditions differ; do not compare directly."
        else:
            reason = "Insufficient structured evidence to establish five-axis comparability."
        assessments.append(ComparabilityAssessment(
            paper_ids=[left.paper_id, right.paper_id], task_same=task,
            dataset_split_same=datasets, metric_same=metrics,
            scale_budget_similar=settings, conditions_comparable=settings,
            comparable=comparable, reason=reason,
        ))
    return assessments


def _overlap(left, right) -> bool | None:
    if not left or not right:
        return None
    left_tokens = _terms(" ".join(item.text for item in left))
    right_tokens = _terms(" ".join(item.text for item in right))
    return bool(left_tokens & right_tokens)


def _terms(text: str) -> set[str]:
    import re
    ignored = {"the", "and", "with", "using", "method", "model", "results", "on", "for"}
    return {term for term in re.findall(r"[a-z0-9][a-z0-9._-]+", text.lower()) if term not in ignored}


def find_contradictions(
    facts: list[StructuredFacts], assessments: list[ComparabilityAssessment] | None = None
) -> list[Conflict]:
    """Flag outcome language that points in opposite directions across papers."""
    positive_terms = ("improve", "outperform", "achieve")
    negative_terms = ("fail", "cannot", "worse", "degrade")
    positives = [
        (item.paper_id, fact)
        for item in facts
        for fact in item.conclusions
        if any(term in fact.text.lower() for term in positive_terms)
    ]
    negatives = [
        (item.paper_id, fact)
        for item in facts
        for fact in item.conclusions
        if any(term in fact.text.lower() for term in negative_terms)
    ]
    conflicts: list[Conflict] = []
    comparable_pairs = {
        frozenset(item.paper_ids) for item in (assessments or assess_comparability(facts)) if item.comparable
    }
    for positive_paper_id, positive_fact in positives:
        opposing = [
            negative_fact
            for negative_paper_id, negative_fact in negatives
            if frozenset((positive_paper_id, negative_paper_id)) in comparable_pairs
        ]
        if not opposing:
            continue
        conflicts.append(
            Conflict(
                message=(
                    f"{positive_paper_id} reports positive outcome language while other candidate papers "
                    "report negative outcome language; inspect their experimental settings."
                ),
                positive_evidence_ids=[positive_fact.evidence_id],
                negative_evidence_ids=[fact.evidence_id for fact in opposing],
            )
        )
    return conflicts
