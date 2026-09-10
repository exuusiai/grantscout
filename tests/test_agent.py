from pathlib import Path

from paperscout.agent.loop import PaperScoutAgent
from paperscout.agent.planner import decompose_question
from paperscout.config import Settings
from paperscout.models.schemas import EvidenceItem
from paperscout.retrieval.parser import parse_document
from paperscout.retrieval.store import CorpusStore
from paperscout.tools.evidence import extract_structured_facts


class FakeJsonModel:
    def __init__(self, payload):
        self.payload = payload

    def chat_json(self, **kwargs):
        return self.payload


def test_model_planner_returns_bounded_questions() -> None:
    questions = decompose_question(
        "Which methods improve evidence recall?",
        model_client=FakeJsonModel({"sub_questions": ["methods", "datasets", "methods"]}),
    )

    assert questions == ["methods", "datasets"]


def test_model_fact_extraction_keeps_only_known_evidence_ids() -> None:
    evidence = EvidenceItem(
        id="paper:section:0000:evidence:0000",
        paper_id="paper",
        section_id="paper:section:0000",
        text="We introduce a grounded retrieval method.",
    )
    facts = extract_structured_facts(
        None,
        "paper",
        [evidence],
        model_client=FakeJsonModel(
            {
                "methods": [
                    {"text": evidence.text, "evidence_id": evidence.id},
                    {"text": "Unsupported claim", "evidence_id": "missing"},
                ]
            }
        ),
    )

    assert [fact.evidence_id for fact in facts.methods] == [evidence.id]


def test_agent_generates_traceable_report(tmp_path: Path) -> None:
    source = tmp_path / "paper.txt"
    source.write_text(
        "Methods\n\nWe introduce a grounded retrieval method.\n\n"
        "Results\n\nOur method improves evidence recall on the benchmark dataset.\n\n"
        "Limitations\n\nHowever, it fails when evidence is missing.\n",
        encoding="utf-8",
    )
    settings = Settings(
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        max_steps=20,
        max_tool_calls=20,
    )
    settings.prepare_directories()
    document = parse_document(source, paper_id="traceable-paper")
    with CorpusStore(tmp_path / "corpus.sqlite") as store:
        store.upsert(document)
        state = PaperScoutAgent(store, settings).run("Which retrieval method improves evidence recall?")

    assert state.status == "completed"
    assert state.selected_papers
    assert state.evidence_items
    assert state.claims
    assert all(claim.support_status == "supported" for claim in state.claims)
    assert state.citation_audit is not None
    assert list((tmp_path / "runs").glob("*.json"))
    assert list((tmp_path / "runs").glob("*.html"))
