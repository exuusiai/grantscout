import json
from pathlib import Path

import pytest

from paperscout.agent.loop import PaperScoutAgent
from paperscout.config import Settings
from paperscout.evaluation.datasets import evaluation_records
from paperscout.models.schemas import Claim, Fact, StructuredFacts
from paperscout.retrieval.parser import parse_document
from paperscout.retrieval.store import CorpusStore
from paperscout.tools.audit import audit_citations
from paperscout.tools.comparison import find_contradictions


def test_no_match_retains_missing_evidence_warning(tmp_path: Path) -> None:
    source = tmp_path / "paper.txt"
    source.write_text("Methods\n\nGrounded retrieval uses citations.", encoding="utf-8")
    settings = Settings(
        runs_dir=tmp_path / "runs",
        retrieval_mode="lexical",
        use_model_reasoning=False,
    )
    with CorpusStore(tmp_path / "corpus.sqlite") as store:
        store.upsert(parse_document(source, paper_id="grounded"))
        state = PaperScoutAgent(store, settings).run("What causes marine coral bleaching?")

    assert state.selected_papers == []
    assert "No local papers matched the research question." in state.warnings


def test_missing_citation_is_audit_failure() -> None:
    audit = audit_citations([Claim(id="claim:0000", text="Unsupported", evidence_ids=[])], [])

    assert audit.status == "failed"
    assert audit.items[0].reason == "No resolvable evidence IDs were attached to this claim."


def test_unsupported_document_extension_fails_explicitly(tmp_path: Path) -> None:
    source = tmp_path / "paper.html"
    source.write_text("<p>not an accepted paper input</p>", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported paper format"):
        parse_document(source)


def test_cross_paper_conflict_keeps_both_evidence_ids() -> None:
    positive = Fact(
        field="conclusion",
        text="The method improves evidence recall.",
        evidence_id="positive:section:0001:evidence:0000",
    )
    negative = Fact(
        field="conclusion",
        text="The method fails on the benchmark.",
        evidence_id="negative:section:0001:evidence:0000",
    )

    conflicts = find_contradictions(
        [
            StructuredFacts(paper_id="positive", conclusions=[positive]),
            StructuredFacts(paper_id="negative", conclusions=[negative]),
        ]
    )

    assert conflicts[0].positive_evidence_ids == [positive.evidence_id]
    assert conflicts[0].negative_evidence_ids == [negative.evidence_id]


def test_evaluation_query_without_query_field_fails_explicitly(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps({"relevant_paper_ids": ["paper-1"]}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="require a query field"):
        evaluation_records(path)
