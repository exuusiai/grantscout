from itertools import combinations

from grantscout.models.schemas import ComparabilityAssessment, Conflict, StructuredFacts


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
    # Pair the evidence-rich papers first. Twenty papers otherwise produce 190
    # mostly unknown comparisons and obscure the few defensible judgments.
    ranked = sorted(facts, key=_evidence_richness, reverse=True)[:8]
    for left, right in combinations(ranked, 2):
        datasets = _axis_overlap(left.datasets, right.datasets)
        metrics = _axis_overlap(left.metrics, right.metrics)
        settings = _axis_overlap(left.experimental_settings, right.experimental_settings)
        task = _task_overlap(left, right)
        known = [task, datasets, metrics, settings]
        comparable = all(value is True for value in (task, datasets, metrics)) and settings is not False
        if comparable:
            status = "comparable"
            reason = "Same task, dataset, and metric are evidenced; experimental conditions do not conflict."
        elif any(value is False for value in known):
            status = "condition_mismatch"
            reason = "Key task, dataset, metric, or experimental conditions differ; do not compare directly."
        else:
            status = "insufficient_evidence"
            reason = "Insufficient structured evidence to establish five-axis comparability."
        assessments.append(ComparabilityAssessment(
            paper_ids=[left.paper_id, right.paper_id], task_same=task,
            dataset_split_same=datasets, metric_same=metrics,
            scale_budget_similar=settings, conditions_comparable=settings,
            comparable=comparable, status=status,
            evidence_ids=_evidence_ids(left, right), reason=reason,
        ))
    return assessments


def _evidence_richness(item: StructuredFacts) -> int:
    return sum(len(getattr(item, field)) for field in (
        "methods", "datasets", "experimental_settings", "metrics", "conclusions"
    ))


def _evidence_ids(*items: StructuredFacts) -> list[str]:
    ids = []
    for item in items:
        for field in ("methods", "datasets", "experimental_settings", "metrics", "conclusions"):
            for fact in getattr(item, field):
                if fact.evidence_id not in ids:
                    ids.append(fact.evidence_id)
    return ids[:12]


def _task_overlap(left: StructuredFacts, right: StructuredFacts) -> bool | None:
    # Conclusions often contain the task more explicitly than method names.
    left_facts = [*left.conclusions, *left.methods]
    right_facts = [*right.conclusions, *right.methods]
    return _axis_overlap(left_facts, right_facts)


def _axis_overlap(left, right) -> bool | None:
    if not left or not right:
        return None
    left_terms = _terms(" ".join(item.text for item in left))
    right_terms = _terms(" ".join(item.text for item in right))
    if not left_terms or not right_terms:
        return None
    shared = left_terms & right_terms
    return len(shared) >= 1


def _overlap(left, right) -> bool | None:
    return _axis_overlap(left, right)


def _terms(text: str) -> set[str]:
    import re
    ignored = {
        "the", "and", "with", "using", "method", "model", "results", "result",
        "on", "for", "from", "this", "that", "our", "we", "show", "paper",
        "approach", "performance", "dataset", "benchmark", "split", "train",
        "training", "test", "testing", "validation", "evaluation", "set",
    }
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
