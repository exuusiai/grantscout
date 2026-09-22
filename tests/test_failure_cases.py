import json
from pathlib import Path

import pytest

from grantscout.agent.loop import GrantScoutAgent
from grantscout.config import Settings
from grantscout.evaluation.datasets import evaluation_records
from grantscout.models.schemas import Claim, Fact, StructuredFacts
from grantscout.retrieval.parser import parse_document
from grantscout.retrieval.store import CorpusStore
from grantscout.tools.audit import audit_citations
from grantscout.tools.comparison import assess_comparability, find_contradictions


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
        state = GrantScoutAgent(store, settings).run("What causes marine coral bleaching?")

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
    shared_dataset = Fact(field="dataset", text="SciFact benchmark", evidence_id="dataset")
    shared_metric = Fact(field="metric", text="evidence recall", evidence_id="metric")
    shared_method = Fact(field="method", text="retrieval method", evidence_id="method")

    conflicts = find_contradictions(
        [
            StructuredFacts(paper_id="positive", conclusions=[positive], datasets=[shared_dataset], metrics=[shared_metric], methods=[shared_method]),
            StructuredFacts(paper_id="negative", conclusions=[negative], datasets=[shared_dataset], metrics=[shared_metric], methods=[shared_method]),
        ]
    )

    assert conflicts[0].positive_evidence_ids == [positive.evidence_id]
    assert conflicts[0].negative_evidence_ids == [negative.evidence_id]


def test_cross_paper_conflict_is_suppressed_without_comparability() -> None:
    positive = Fact(field="conclusion", text="The method improves accuracy.", evidence_id="positive")
    negative = Fact(field="conclusion", text="The method fails badly.", evidence_id="negative")

    assert find_contradictions([
        StructuredFacts(paper_id="a", conclusions=[positive]),
        StructuredFacts(paper_id="b", conclusions=[negative]),
    ]) == []


def test_comparability_distinguishes_mismatch_from_missing_evidence() -> None:
    common_method = Fact(field="method", text="group relative policy optimization", evidence_id="method")
    common_metric = Fact(field="metric", text="reward accuracy", evidence_id="metric")
    left_dataset = Fact(field="dataset", text="GSM8K test split", evidence_id="gsm8k")
    right_dataset = Fact(field="dataset", text="MATH validation split", evidence_id="math")
    assessments = assess_comparability([
        StructuredFacts(paper_id="a", methods=[common_method], metrics=[common_metric], datasets=[left_dataset]),
        StructuredFacts(paper_id="b", methods=[common_method], metrics=[common_metric], datasets=[right_dataset]),
    ])

    assert assessments[0].status == "condition_mismatch"
    assert assessments[0].comparable is False
    assert {"method", "metric", "gsm8k", "math"}.issubset(assessments[0].evidence_ids)

    missing = assess_comparability([
        StructuredFacts(paper_id="a", methods=[common_method]),
        StructuredFacts(paper_id="b", methods=[common_method]),
    ])
    assert missing[0].status == "insufficient_evidence"


def test_evaluation_query_without_query_field_fails_explicitly(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps({"relevant_paper_ids": ["paper-1"]}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="require a query field"):
        evaluation_records(path)
