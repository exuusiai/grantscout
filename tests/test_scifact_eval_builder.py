import importlib.util
import json
from pathlib import Path

from grantscout.retrieval.parser import build_document
from grantscout.retrieval.store import CorpusStore
from grantscout.models.schemas import Paper


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_scifact_eval.py"
SPEC = importlib.util.spec_from_file_location("build_scifact_eval", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_build_scifact_records_maps_labeled_sentence_to_evidence(tmp_path: Path) -> None:
    raw_corpus = tmp_path / "corpus.jsonl"
    raw_corpus.write_text(
        json.dumps(
            {
                "doc_id": 42,
                "title": "Example",
                "abstract": ["The method improves recall.", "It has a limitation."],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus.sqlite"
    with CorpusStore(corpus) as store:
        store.upsert(
            build_document(
                Paper(id="42", title="Example"),
                [("Abstract", "The method improves recall.\nIt has a limitation.", None)],
            )
        )
    claims = tmp_path / "claims.jsonl"
    claims.write_text(
        json.dumps(
            {
                "id": 1,
                "claim": "The method improves recall.",
                "cited_doc_ids": [42],
                "evidence": {"42": [{"sentences": [0], "label": "SUPPORT"}]},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = builder.build_records(claims, raw_corpus, corpus)

    assert records[0]["relevant_paper_ids"] == ["42"]
    assert records[0]["relevant_evidence_ids"] == ["42:section:0000:evidence:0000"]
    assert "conflict_expected" not in records[0]
