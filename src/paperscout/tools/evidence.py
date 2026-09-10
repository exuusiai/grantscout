import json
import re

from paperscout.models.llm import ModelClientError
from paperscout.models.schemas import Claim, EvidenceItem, Fact, StructuredFacts
from paperscout.retrieval.store import CorpusStore


_FIELD_KEYWORDS: dict[str, tuple[str, ...]] = {
    "method": ("method", "model", "approach", "framework", "algorithm", "we propose", "we introduce"),
    "dataset": ("dataset", "benchmark", "corpus", "data set", "evaluated on", "tested on"),
    "experimental_setting": (
        "experimental setting",
        "experimental setup",
        "configuration",
        "setting",
        "we use",
        "we evaluate",
    ),
    "metric": ("accuracy", "recall", "precision", "f1", "bleu", "rouge", "latency", "score", "%"),
    "conclusion": ("result", "outperform", "improve", "improvement", "show that", "achieve", "conclude"),
    "limitation": ("limitation", "limitations", "future work", "cannot", "fails", "failure", "however"),
}


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?。！？])\s+", text) if sentence.strip()]


def extract_structured_facts(
    store: CorpusStore,
    paper_id: str,
    evidence_items: list[EvidenceItem] | None = None,
    model_client=None,
) -> StructuredFacts:
    """Extract traceable facts, optionally using the configured local model."""
    items = evidence_items or []
    if not items:
        rows = store.search(paper_id, top_k=100)
        items = [row.evidence for row in rows if row.evidence.paper_id == paper_id]
    if model_client is not None:
        return _extract_with_model(paper_id, items, model_client)
    return _extract_with_keywords(paper_id, items)


def _extract_with_keywords(paper_id: str, items: list[EvidenceItem]) -> StructuredFacts:
    """Extract candidate facts using keyword heuristics for the offline baseline."""
    facts = StructuredFacts(paper_id=paper_id)
    seen: set[tuple[str, str]] = set()
    for evidence in items:
        for sentence in _sentences(evidence.text):
            lowered = sentence.lower()
            for field, keywords in _FIELD_KEYWORDS.items():
                if any(keyword in lowered for keyword in keywords):
                    key = (field, sentence)
                    if key in seen:
                        continue
                    seen.add(key)
                    fact = Fact(text=sentence, evidence_id=evidence.id, field=field)  # type: ignore[arg-type]
                    getattr(facts, f"{field}s").append(fact)
    return facts


def _extract_with_model(paper_id: str, items: list[EvidenceItem], model_client) -> StructuredFacts:
    if not items:
        raise ModelClientError("Cannot extract model facts without retrieved evidence")
    payload = model_client.chat_json(
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract only evidence-grounded facts from supplied paper evidence. "
                    "Return JSON arrays named methods, datasets, experimental_settings, "
                    "metrics, conclusions, and limitations. Each item must contain text and "
                    "an evidence_id copied exactly from the input. Never invent evidence IDs."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "paper_id": paper_id,
                        "evidence": [{"evidence_id": item.id, "text": item.text} for item in items],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        max_tokens=1536,
        temperature=0.0,
    )
    if not isinstance(payload, dict):
        raise ModelClientError("Model fact extraction did not return an object")
    evidence_by_id = {item.id: item for item in items}
    result = StructuredFacts(paper_id=paper_id)
    fields = (
        "methods",
        "datasets",
        "experimental_settings",
        "metrics",
        "conclusions",
        "limitations",
    )
    for field in fields:
        values = payload.get(field, [])
        if not isinstance(values, list):
            raise ModelClientError(f"Model fact extraction field {field} is not an array")
        for value in values:
            if not isinstance(value, dict):
                continue
            try:
                fact = Fact.model_validate({**value, "field": field.rstrip("s")})
            except (TypeError, ValueError):
                continue
            evidence = evidence_by_id.get(fact.evidence_id)
            if evidence is None or _text_overlap(fact.text, evidence.text) < 0.2:
                continue
            target = getattr(result, field)
            if not any(
                existing.text == fact.text and existing.evidence_id == fact.evidence_id
                for existing in target
            ):
                target.append(fact)
    if not any(getattr(result, field) for field in fields):
        raise ModelClientError("Model fact extraction returned no grounded facts")
    return result


def _text_overlap(left: str, right: str) -> float:
    left_terms = {term for term in re.findall(r"[\w]+", left.lower()) if len(term) > 2}
    right_terms = {term for term in re.findall(r"[\w]+", right.lower()) if len(term) > 2}
    return len(left_terms & right_terms) / max(len(left_terms), 1)


def verify_claim(claim: Claim, evidence_items: list[EvidenceItem]) -> Claim:
    evidence_text = " ".join(item.text.lower() for item in evidence_items)
    claim_terms = {term for term in re.findall(r"[\w]+", claim.text.lower()) if len(term) > 2}
    matched_terms = {term for term in claim_terms if term in evidence_text}
    if not claim.evidence_ids:
        return claim.model_copy(update={"support_status": "insufficient", "confidence": 0.0})
    if not evidence_items:
        return claim.model_copy(update={"support_status": "unknown", "confidence": 0.0})
    overlap = len(matched_terms) / max(len(claim_terms), 1)
    status = "supported" if overlap >= 0.35 else "insufficient"
    return claim.model_copy(update={"support_status": status, "confidence": round(min(overlap, 1.0), 3)})
