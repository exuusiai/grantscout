from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class PaperSection(BaseModel):
    id: str
    paper_id: str
    title: str
    section_index: int = Field(ge=0)
    text: str
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)


class Paper(BaseModel):
    id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = Field(default=None, ge=0, le=3000)
    abstract: str = ""
    source_path: str | None = None


class EvidenceItem(BaseModel):
    id: str
    paper_id: str
    section_id: str
    text: str
    page: int | None = Field(default=None, ge=1)
    start_char: int = Field(default=0, ge=0)
    end_char: int = Field(default=0, ge=0)


class ParsedPaper(BaseModel):
    paper: Paper
    sections: list[PaperSection] = Field(default_factory=list)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)


class SearchResult(BaseModel):
    evidence: EvidenceItem
    paper: Paper
    score: float = Field(ge=0)
    matched_terms: list[str] = Field(default_factory=list)


class PaperCandidate(BaseModel):
    paper: Paper
    score: float = Field(ge=0)
    matched_terms: list[str] = Field(default_factory=list)
    relevance_reason: str


class Fact(BaseModel):
    text: str
    localized_text: str | None = None
    evidence_id: str
    field: Literal[
        "method",
        "dataset",
        "experimental_setting",
        "metric",
        "conclusion",
        "limitation",
    ]


class StructuredFacts(BaseModel):
    paper_id: str
    methods: list[Fact] = Field(default_factory=list)
    datasets: list[Fact] = Field(default_factory=list)
    experimental_settings: list[Fact] = Field(default_factory=list)
    metrics: list[Fact] = Field(default_factory=list)
    conclusions: list[Fact] = Field(default_factory=list)
    limitations: list[Fact] = Field(default_factory=list)


class Claim(BaseModel):
    id: str
    text: str
    localized_text: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    support_status: Literal["supported", "refuted", "insufficient", "unknown"] = "unknown"
    confidence: float = Field(default=0.0, ge=0, le=1)


class CitationAuditItem(BaseModel):
    claim_id: str
    supported: bool
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str


class CitationAudit(BaseModel):
    status: Literal["passed", "warning", "failed"]
    supported_claims: int = 0
    unsupported_claims: int = 0
    items: list[CitationAuditItem] = Field(default_factory=list)


class Conflict(BaseModel):
    """Evidence-backed incompatible outcome language across different papers."""

    message: str
    positive_evidence_ids: list[str] = Field(default_factory=list)
    negative_evidence_ids: list[str] = Field(default_factory=list)


class ToolCall(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tool: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    status: Literal["ok", "error"] = "ok"
    duration_ms: int = Field(default=0, ge=0)
    retryable: bool = False


class BudgetState(BaseModel):
    steps: int = 0
    tool_calls: int = 0
    papers: int = 0


class ResearchState(BaseModel):
    run_id: str
    question: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    sub_questions: list[str] = Field(default_factory=list)
    candidate_papers: list[PaperCandidate] = Field(default_factory=list)
    selected_papers: list[Paper] = Field(default_factory=list)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    facts: list[StructuredFacts] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    comparison: list[dict[str, Any]] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    citation_audit: CitationAudit | None = None
    warnings: list[str] = Field(default_factory=list)
    tool_history: list[ToolCall] = Field(default_factory=list)
    budget: BudgetState = Field(default_factory=BudgetState)
    status: Literal["running", "completed", "failed"] = "running"
