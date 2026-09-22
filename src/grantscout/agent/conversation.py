from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Literal

from pydantic import BaseModel, Field

from grantscout.config import Settings
from grantscout.models.llm import ModelClientError, OpenAICompatibleClient, strip_thinking
from grantscout.models.schemas import ResearchConstraints
from grantscout.retrieval.store import CorpusStore


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatCitation(BaseModel):
    evidence_id: str
    paper_title: str
    snippet: str
    scope: str = "public"


class ConversationResult(BaseModel):
    status: Literal["clarification", "ready"]
    message: str
    refined_question: str | None = None
    ranking: Literal["relevance", "recent", "citations"] = "relevance"
    conversation_id: str | None = None
    constraints: ResearchConstraints = Field(default_factory=ResearchConstraints)
    paper_limit: int = Field(default=5, ge=1, le=20)
    citations: list[ChatCitation] = Field(default_factory=list)


def understand_request(
    messages: list[ConversationMessage],
    settings: Settings,
    locale: str = "zh",
    memory: list[str] | None = None,
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
    system_content = (
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
    )
    if memory:
        system_content += (
            "\n\nKnown project memory (durable facts and preferences distilled from "
            "past conversations; honor them instead of re-asking):\n"
            + "\n".join(f"- {item}" for item in memory)
        )
    try:
        payload = client.chat_json(
            messages=[
                {"role": "system", "content": system_content},
                *[message.model_dump() for message in messages],
            ],
            max_tokens=420,
            temperature=0.0,
        )
        return _validate_result(payload)
    except (ModelClientError, ValueError, TypeError):
        return _fallback(messages[-1].content, locale)


def remember_conversation(messages: list[ConversationMessage], settings: Settings) -> list[str]:
    """Distill durable project facts from a finished conversation (model-backed).

    Returns an empty list when the model is disabled or extraction fails —
    memory must never break the chat itself.
    """
    if not settings.use_model_reasoning or not messages:
        return []
    try:
        client = OpenAICompatibleClient(
            settings.model_base_url,
            settings.model_api_key,
            settings.model_name,
            timeout_seconds=min(settings.model_timeout_seconds, 45.0),
        )
        payload = client.chat_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Read this conversation and extract 0-3 durable facts or "
                        "preferences about the user's research project worth remembering "
                        "for future conversations (topic, constraints, naming, data, "
                        "preferred methods). Skip one-off chit-chat and anything already "
                        "stale. Never invent facts. Return JSON only: "
                        '{"memories": ["..."]}; an empty array when nothing is durable.'
                    ),
                },
                *[message.model_dump() for message in messages[-8:]],
            ],
            max_tokens=300,
            temperature=0.0,
        )
        if not isinstance(payload, dict):
            return []
        memories = payload.get("memories", [])
        if not isinstance(memories, list):
            return []
        return [item.strip()[:300] for item in memories if isinstance(item, str) and item.strip()][:3]
    except (ModelClientError, ValueError, TypeError):
        return []


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


def _project_client(settings: Settings) -> OpenAICompatibleClient | None:
    if not settings.use_model_reasoning:
        return None
    return OpenAICompatibleClient(
        settings.model_base_url,
        settings.model_api_key,
        settings.model_name,
        timeout_seconds=min(settings.model_timeout_seconds, 45.0),
    )


def answer_with_project_sources(
    question: str,
    store: CorpusStore,
    settings: Settings,
    locale: str = "zh",
    top_k: int = 4,
) -> tuple[str | None, list[ChatCitation]]:
    """Ground a chat reply in project-corpus evidence; never invents sources.

    Returns (answer, citations). ``answer`` is None when the model is off —
    callers keep their original message and still receive the citations.
    """
    results = store.search(question, top_k=top_k) if question.strip() else []
    if not results and question.strip():
        # CJK 词法检索在 unicode61 分词下偏弱;trigram 句子索引按子串兜底。
        from grantscout.retrieval.sentences import SentenceIndex

        results = [
            SimpleNamespace(evidence=hit.evidence, paper=None)
            for hit in SentenceIndex(store).continue_fragment(question, top_k=top_k)
        ]
    if not results:
        return None, []

    def evidence_of(item):
        return item.evidence

    def title_of(item):
        return item.paper.title if item.paper is not None else (
            store.get_paper(item.evidence.paper_id).title
            if store.get_paper(item.evidence.paper_id) else item.evidence.paper_id
        )

    citations = [
        ChatCitation(
            evidence_id=item.evidence.id,
            paper_title=title_of(item),
            snippet=item.evidence.text[:200],
            scope=store.get_paper_meta(item.evidence.paper_id).get("scope") or "public",
        )
        for item in results
    ]
    client = _project_client(settings)
    if client is None:
        return None, citations
    sources = "\n".join(
        f"【文献{index}】({title_of(item)}) {item.evidence.text}"
        for index, item in enumerate(results, start=1)
    )
    language = "用中文回答" if locale == "zh" else "Answer in English"
    try:
        payload = client.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a research assistant grounded in the user's project library. {language}。"
                        "只依据提供的资料回答;使用资料的句子末尾标注【文献n】;资料不足时明确说明,"
                        "不要编造。直接输出回答正文。"
                    ),
                },
                {"role": "user", "content": f"问题:{question}\n\n资料:\n{sources}"},
            ],
            max_tokens=768,
            temperature=0.2,
        ).content
    except ModelClientError:
        return None, citations
    answer = strip_thinking(payload).strip()
    if not answer:
        return None, citations
    answer = re.sub(r"【文献(\d{1,2})】", _keep_valid_marker(len(results)), answer)
    used, seen = [], set()
    for match in re.finditer(r"【文献(\d{1,2})】", answer):
        index = int(match.group(1))
        if index in seen or not 1 <= index <= len(results):
            continue
        seen.add(index)
        used.append(citations[index - 1])
    return answer, (used or citations)



def _keep_valid_marker(source_count: int):
    def replace(match: re.Match[str]) -> str:
        return match.group(0) if 1 <= int(match.group(1)) <= source_count else ""

    return replace
