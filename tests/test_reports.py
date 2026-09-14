from datetime import UTC, datetime

from paperscout.models.schemas import (
    Claim,
    ComparabilityAssessment,
    EvidenceItem,
    Paper,
    ResearchDecision,
    ResearchState,
)
from paperscout.reports.renderer import render_html, render_markdown


def test_reports_are_localized_and_omit_internal_audit_sections() -> None:
    evidence_id = "paper-1:section:0001:evidence:0000"
    state = ResearchState(
        run_id="test-run",
        question="What improves evidence recall?",
        finished_at=datetime.now(UTC),
        selected_papers=[Paper(id="paper-1", title="Evidence Study")],
        evidence_items=[
            EvidenceItem(
                id=evidence_id,
                paper_id="paper-1",
                section_id="paper-1:section:0001",
                text="The method improves evidence recall.",
            )
        ],
        claims=[
            Claim(
                id="claim:0000",
                text="The method improves evidence recall.",
                localized_text="该方法提高了证据召回率。",
                evidence_ids=[evidence_id],
            )
        ],
        comparability=[
            ComparabilityAssessment(
                paper_ids=["paper-1", "paper-2"],
                status="condition_mismatch",
                reason="Different training budgets",
            )
        ],
        decisions=[
            ResearchDecision(
                paper_id="paper-1",
                recommendation="read",
                readiness_score=40,
                reason="Score reflects only evidence recovered from the paper; unknown repository and hardware fields are penalized.",
                missing_information=["metric definition"],
            )
        ],
    )

    markdown = render_markdown(state)
    html = render_html(state)

    assert "## Executive Summary" in markdown
    assert "## Dataset and Experimental Setup Comparison" in markdown
    assert "## Open Questions" in markdown
    assert "Evidence Table" not in markdown
    assert "Citation Audit" not in markdown
    assert "Run Metadata" not in markdown
    assert f'href="#evidence-{evidence_id}"' not in html
    assert "该方法提高了证据召回率。" in render_html(state, "zh")
    assert "该方法提高了证据召回率。" not in render_html(state, "en")
    chinese = render_html(state, "zh")
    assert "条件不一致，不可直接比较" in chinese
    assert "建议阅读" in chinese
    assert "评分仅依据论文中已恢复的证据" in chinese
    assert "尚缺：指标定义" in chinese
    assert "Condition mismatch" not in chinese
    assert "Score reflects only evidence" not in chinese
    assert "Missing:" not in chinese
