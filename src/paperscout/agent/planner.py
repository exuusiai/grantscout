import json
import re

from paperscout.models.llm import ModelClientError


def _decompose_deterministically(question: str) -> list[str]:
    """Create deterministic sub-questions for the offline baseline."""
    normalized = " ".join(question.split())
    parts = [part.strip(" ，,；;。") for part in re.split(r"\s+(?:and|or)\s+|[；;]", normalized) if part.strip()]
    sub_questions: list[str] = []
    for part in parts:
        if part not in sub_questions:
            sub_questions.append(part)
    if len(sub_questions) > 1:
        context = sub_questions[0]
        sub_questions = [
            part if index == 0 or len(part.split()) >= 5 else f"{context}; {part}"
            for index, part in enumerate(sub_questions)
        ]
    if len(sub_questions) == 1:
        sub_questions.extend(
            [
                f"{normalized} datasets and experimental settings",
                f"{normalized} limitations and failure cases",
            ]
        )
    return sub_questions[:5]


def decompose_question(question: str, model_client=None) -> list[str]:
    """Use the local model when available, with strict deterministic fallback."""
    if model_client is None:
        return _decompose_deterministically(question)
    payload = model_client.chat_json(
        messages=[
            {
                "role": "system",
                "content": (
                    "Decompose a scientific literature review question. Return only JSON "
                    "with a sub_questions array containing 1 to 5 concise search questions."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"question": question}, ensure_ascii=False),
            },
        ],
        max_tokens=512,
        temperature=0.0,
    )
    values = payload.get("sub_questions") if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ModelClientError("Model question decomposition did not return sub_questions")
    questions: list[str] = [" ".join(question.split())]
    anchor_terms = {
        term.lower() for term in re.findall(r"[A-Za-z][A-Za-z0-9+._-]{1,}", question)
    }
    for value in values:
        if isinstance(value, str) and value.strip():
            normalized = " ".join(value.split())
            if anchor_terms and not anchor_terms.intersection(
                term.lower() for term in re.findall(r"[A-Za-z][A-Za-z0-9+._-]{1,}", normalized)
            ):
                continue
            if normalized not in questions:
                questions.append(normalized)
    if not questions:
        raise ModelClientError("Model question decomposition returned no usable sub-questions")
    return questions[:3]
