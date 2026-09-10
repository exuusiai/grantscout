from paperscout.models.schemas import CitationAudit, CitationAuditItem, Claim, EvidenceItem
from paperscout.tools.evidence import verify_claim


def audit_citations(claims: list[Claim], evidence_items: list[EvidenceItem]) -> CitationAudit:
    evidence_by_id = {item.id: item for item in evidence_items}
    items: list[CitationAuditItem] = []
    for claim in claims:
        cited_evidence = [evidence_by_id[item_id] for item_id in claim.evidence_ids if item_id in evidence_by_id]
        checked_claim = verify_claim(claim, cited_evidence)
        claim.support_status = checked_claim.support_status
        claim.confidence = checked_claim.confidence
        supported = checked_claim.support_status == "supported"
        reason = (
            f"Evidence overlap confidence {checked_claim.confidence:.3f}."
            if cited_evidence
            else "No resolvable evidence IDs were attached to this claim."
        )
        items.append(
            CitationAuditItem(
                claim_id=claim.id,
                supported=supported,
                evidence_ids=claim.evidence_ids,
                reason=reason,
            )
        )
    supported_count = sum(item.supported for item in items)
    unsupported_count = len(items) - supported_count
    status = "passed" if unsupported_count == 0 else "warning" if supported_count else "failed"
    return CitationAudit(
        status=status,
        supported_claims=supported_count,
        unsupported_claims=unsupported_count,
        items=items,
    )
