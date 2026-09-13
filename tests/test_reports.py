from datetime import UTC, datetime

from paperscout.models.schemas import Claim, EvidenceItem, Paper, ResearchState
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
