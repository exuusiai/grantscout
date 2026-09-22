"""Data models for template-based proposal drafting."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from grantscout.models.schemas import ResearchConstraints, ResearchState, ToolCall


class SectionSpec(BaseModel):
    """One template section with writing guidance and a word budget."""

    key: str
    title: str
    guidance: str = ""
    min_words: int = Field(default=0, ge=0)
    max_words: int = Field(default=0, ge=0)
    requires_citations: bool = True
    structured_indicators: bool = False
    generatable: bool = True


class IndicatorItem(BaseModel):
    """One assessment indicator split into the six review-critical fields."""

    name: str
    target_value: str = ""
    test_conditions: str = ""
    test_method: str = ""
    acceptance_materials: str = ""
    source_basis: str = ""


class ProposalTemplate(BaseModel):
    id: str
    name: str
    description: str = ""
    language: Literal["zh", "en"] = "zh"
    sections: list[SectionSpec] = Field(min_length=1)


class IdeaBrief(BaseModel):
    """The clarified topic the proposal is written around."""

    topic: str
    goals: list[str] = Field(default_factory=list)
    innovation_points: list[str] = Field(default_factory=list)
    domain: str | None = None
    duration_months: int | None = Field(default=None, ge=1, le=120)
    constraints: ResearchConstraints = Field(default_factory=ResearchConstraints)
    open_questions: list[str] = Field(default_factory=list)


class GapStatement(BaseModel):
    """An evidence-backed statement of what existing work has not solved."""

    id: str
    statement: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0, le=1)


class ScienceQuestion(BaseModel):
    """A key scientific question derived from gap statements."""

    id: str
    question: str
    gap_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class LiteratureDossier(BaseModel):
    """Literature evidence for the proposal, reusing the research state."""

    topic: str
    state: ResearchState
    gaps: list[GapStatement] = Field(default_factory=list)
    science_questions: list[ScienceQuestion] = Field(default_factory=list)


class OutlineSection(BaseModel):
    key: str
    title: str
    bullet_points: list[str] = Field(default_factory=list)
    word_budget: int = Field(default=0, ge=0)
    evidence_ids: list[str] = Field(default_factory=list)


class Outline(BaseModel):
    template_id: str
    sections: list[OutlineSection] = Field(default_factory=list)
    approved: bool = False


class DraftCitation(BaseModel):
    """One 【文献n】 marker inside a section draft, resolved to evidence."""

    marker: str
    evidence_id: str
    paper_title: str
    page: int | None = None


class SectionDraft(BaseModel):
    """One version of one proposal section."""

    section_key: str
    version: int = 1
    content_md: str
    citations: list[DraftCitation] = Field(default_factory=list)
    indicators: list[IndicatorItem] = Field(default_factory=list)
    source: Literal["agent", "material"] = "agent"
    warnings: list[str] = Field(default_factory=list)
    word_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ProposalReview(BaseModel):
    """Deterministic self-check results across all drafted sections."""

    status: Literal["passed", "warning", "failed"] = "warning"
    total_citations: int = 0
    unverified_citations: int = 0
    sections_missing_evidence: list[str] = Field(default_factory=list)
    word_counts: dict[str, int] = Field(default_factory=dict)
    budget_warnings: list[str] = Field(default_factory=list)


class ProposalDocument(BaseModel):
    id: str
    title: str
    template_id: str
    locale: Literal["zh", "en"] = "zh"
    corpus_path: str | None = None
    idea_brief: IdeaBrief
    dossier: LiteratureDossier | None = None
    outline: Outline | None = None
    sections: dict[str, list[SectionDraft]] = Field(default_factory=dict)
    review: ProposalReview | None = None
    warnings: list[str] = Field(default_factory=list)
    tool_history: list[ToolCall] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: Literal["drafting", "completed", "failed"] = "drafting"

    def latest_draft(self, section_key: str) -> SectionDraft | None:
        versions = self.sections.get(section_key) or []
        return versions[-1] if versions else None

    def add_draft(self, draft: SectionDraft) -> None:
        versions = self.sections.setdefault(draft.section_key, [])
        draft.version = len(versions) + 1
        versions.append(draft)
        self.updated_at = datetime.now(UTC)


class ProposalRequest(BaseModel):
    topic: str = Field(min_length=4, max_length=500)
    template_id: str = "project-plan-v1"
    locale: Literal["zh", "en"] = "zh"
    goals: list[str] = Field(default_factory=list)
    duration_months: int | None = Field(default=None, ge=1, le=120)
    constraints: ResearchConstraints = Field(default_factory=ResearchConstraints)
    corpus: str | None = None
