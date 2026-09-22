"""Data models for the inline completion service."""

from typing import Literal

from pydantic import BaseModel, Field


class CompletionCitation(BaseModel):
    evidence_id: str
    paper_id: str
    paper_title: str
    page: int | None = None
    scope: Literal["public", "private"] = "public"
    source_path: str | None = None
    source_url: str | None = None


class CompletionCandidate(BaseModel):
    """One inline suggestion, always carrying its evidence citations."""

    text: str
    kind: Literal["verbatim", "lexical", "semantic", "llm", "command"]
    score: float = Field(ge=0, le=1)
    trigger: str = ""
    citations: list[CompletionCitation] = Field(default_factory=list)


class CompletionRequest(BaseModel):
    prefix: str = Field(default="", max_length=8000)
    suffix: str = Field(default="", max_length=4000)
    session_id: str | None = None
    request_id: str | None = None
    max_candidates: int = Field(default=3, ge=1, le=5)
    corpus: str | None = None
    locale: Literal["zh", "en"] = "zh"
    mode: Literal["auto", "verbatim", "lexical", "semantic", "llm"] = "auto"
    command: Literal["summarize", "refine", "find"] | None = None
    command_text: str = Field(default="", max_length=8000)


class CompletionResponse(BaseModel):
    candidates: list[CompletionCandidate] = Field(default_factory=list)
    filters_applied: dict[str, list[str]] = Field(default_factory=dict)
    filter_warnings: list[str] = Field(default_factory=list)
    cache_hit: bool = False
    superseded: bool = False
    latency_ms: int = 0
