import importlib.util
import json
from pathlib import Path

from paperscout.models.schemas import Paper
from paperscout.retrieval.parser import build_document
from paperscout.retrieval.store import CorpusStore


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_qasper_eval.py"
SPEC = importlib.util.spec_from_file_location("build_qasper_eval", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_qasper_list_sections_and_evidence_annotations_are_mapped(tmp_path: Path) -> None:
    source = tmp_path / "dev.json"
    source.write_text(
        json.dumps(
            {
                "paper_id": "qasper-paper",
                "title": "Qasper Example",
                "full_text": [
                    {"section_name": "Results", "paragraphs": ["The method improves recall.", "It is fast."]}
                ],
                "qas": {
                    "question": ["What improves recall?"],
                    "question_id": ["q1"],
                    "evidence": [[["Results", 0]]],
                    "answer": [[{"answer": "The method", "extractive_spans": ["The method"]}]],
                },
            }
        ),
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus.sqlite"
    with CorpusStore(corpus) as store:
        store.upsert(
            build_document(
                Paper(id="qasper-paper", title="Qasper Example"),
                [("Results", "The method improves recall.\n\nIt is fast.", None)],
            )
        )

    records = builder.build_records(source, corpus)

    assert records[0]["id"] == "q1"
    assert records[0]["relevant_paper_ids"] == ["qasper-paper"]
    assert records[0]["relevant_evidence_ids"] == [
        "qasper-paper:section:0000:evidence:0000"
    ]


def test_qasper_highlighted_text_is_mapped_to_stored_evidence(tmp_path: Path) -> None:
    source = tmp_path / "dev.json"
    source.write_text(
        json.dumps(
            {
                "id": "qasper-text-paper",
                "title": "Qasper Text Example",
                "full_text": {"Results": ["The method improves recall.", "It is fast."]},
                "qas": [
                    {
                        "question": "What improves recall?",
                        "question_id": "q2",
                        "answers": [
                            {
                                "answer": {
                                    "highlighted_evidence": ["The method improves recall."]
                                }
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus.sqlite"
    with CorpusStore(corpus) as store:
        store.upsert(
            build_document(
                Paper(id="qasper-text-paper", title="Qasper Text Example"),
                [("Results", "The method improves recall.\n\nIt is fast.", None)],
            )
        )

    records = builder.build_records(source, corpus)

    assert records[0]["relevant_evidence_ids"] == [
        "qasper-text-paper:section:0000:evidence:0000"
    ]
