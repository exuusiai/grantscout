from paperscout.models.schemas import Conflict, StructuredFacts


def compare_papers(facts: list[StructuredFacts]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in facts:
        rows.append(
            {
                "paper_id": item.paper_id,
                "methods": [fact.text for fact in item.methods[:5]],
                "datasets": [fact.text for fact in item.datasets[:5]],
                "experimental_settings": [fact.text for fact in item.experimental_settings[:5]],
                "metrics": [fact.text for fact in item.metrics[:5]],
                "conclusions": [fact.text for fact in item.conclusions[:5]],
                "limitations": [fact.text for fact in item.limitations[:5]],
            }
        )
    return rows


def find_contradictions(facts: list[StructuredFacts]) -> list[Conflict]:
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
    for positive_paper_id, positive_fact in positives:
        opposing = [
            negative_fact
            for negative_paper_id, negative_fact in negatives
            if negative_paper_id != positive_paper_id
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
