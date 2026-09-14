from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from paperscout.config import Settings
from paperscout.models.llm import ModelClientError, OpenAICompatibleClient
from paperscout.models.schemas import ResearchConstraints


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ConversationResult(BaseModel):
    status: Literal["clarification", "ready"]
    message: str
    refined_question: str | None = None
    ranking: Literal["relevance", "recent", "citations"] = "relevance"
    conversation_id: str | None = None
    constraints: ResearchConstraints = Field(default_factory=ResearchConstraints)
    paper_limit: int = Field(default=5, ge=1, le=20)


def understand_request(
    messages: list[ConversationMessage], settings: Settings, locale: str = "zh"
) -> ConversationResult:
    """Clarify ambiguous literature requests before retrieval starts."""
    if not messages or messages[-1].role != "user":
        raise ValueError("Conversation must end with a user message")
    client = OpenAICompatibleClient(
        settings.model_base_url,
        settings.model_api_key,
        settings.model_name,
        timeout_seconds=min(settings.model_timeout_seconds, 45.0),
    )
    language = "Chinese" if locale == "zh" else "English"
    try:
        payload = client.chat_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the requirements agent for an AI literature research tool. Read "
                        "the full conversation. If any central term, acronym, abbreviation, domain, "
                        "time range, or comparison target has multiple plausible meanings, ask "
                        "exactly one concise clarification question before searching. For example, "
                        "Chinese '基架' may mean traditional system infrastructure or large-model "
                        "system infrastructure. If the request is sufficiently precise, summarize "
                        "the fully resolved literature task. User corrections override earlier "
                        f"assumptions. Reply in {language}. Return JSON only with status "
                        "(clarification or ready), message, refined_question (null until ready), "
                        "ranking (relevance, recent, or citations), and constraints with time_range, "
                        "open_source_only, max_model_size, max_vram_gb, dataset_preference, "
                        "code_required, and priority (quality, speed, cost, balanced). Use null for "
                        "unknown constraints. Also return paper_limit from 1 to 20 when the user "
                        "specifies a count; otherwise use 5. Do not search yet."
                    ),
                },
                *[message.model_dump() for message in messages],
            ],
            max_tokens=420,
            temperature=0.0,
        )
        return _validate_result(payload)
    except (ModelClientError, ValueError, TypeError):
        return _fallback(messages[-1].content, locale)


def _validate_result(payload: object) -> ConversationResult:
    if not isinstance(payload, dict):
        raise ValueError("Conversation model did not return an object")
    result = ConversationResult.model_validate(payload)
    if result.status == "ready" and not result.refined_question:
        raise ValueError("A ready request must include refined_question")
    return result


def _fallback(question: str, locale: str) -> ConversationResult:
    if "基架" in question:
        message = (
            "你指的是传统系统基础架构，还是大模型系统基础架构？"
            if locale == "zh"
            else "Do you mean traditional system infrastructure or LLM system infrastructure?"
        )
        return ConversationResult(status="clarification", message=message)
    text = question.casefold()
    ranking = "relevance"
    if re.search(r"最新|近期|latest|recent|newest", text):
        ranking = "recent"
    elif re.search(r"高引用|经典|影响力|citation|influential", text):
        ranking = "citations"
    message = "已理解任务，可以开始检索。" if locale == "zh" else "Task understood. Ready to search."
    match = re.search(r"(?:调查|查找|分析|阅读)?\s*(\d{1,2})\s*(?:篇|papers?)", question, re.I)
    paper_limit = max(1, min(20, int(match.group(1)))) if match else 5
    return ConversationResult(
        status="ready", message=message, refined_question=question, ranking=ranking,
        paper_limit=paper_limit,
    )
