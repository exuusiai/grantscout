from pathlib import Path

from paperscout.config import Settings
from paperscout.retrieval.parser import parse_document
from paperscout.retrieval.store import CorpusStore
from paperscout.tools.search import retrieve_evidence, search_papers


def _write_sample(path: Path) -> None:
    path.write_text(
        "Title: Retrieval Evidence Study\n\n"
        "Methods\n\nWe propose a retrieval method for grounded generation.\n\n"
        "Datasets\n\nWe evaluate on the Qasper dataset and report recall.\n\n"
        "Results\n\nOur method improves accuracy over the baseline.\n\n"
        "Limitations\n\nHowever, the method fails on very long documents.\n",
        encoding="utf-8",
    )


def test_parse_ingest_and_search(tmp_path: Path) -> None:
    source = tmp_path / "sample.md"
    _write_sample(source)
    document = parse_document(source, paper_id="sample-paper")

    with CorpusStore(tmp_path / "corpus.sqlite") as store:
        store.upsert(document)
        assert store.paper_count() == 1
        results = search_papers(store, "retrieval method accuracy", top_k=3)

    assert results
    assert results[0].paper.id == "sample-paper"
    assert results[0].matched_terms


def test_settings_prepare_directories(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", runs_dir=tmp_path / "runs")
    settings.prepare_directories()
    assert settings.data_dir.is_dir()
    assert settings.runs_dir.is_dir()


def test_fts_index_tracks_upserts(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    first.write_text("Results\n\nAlpha retrieval improves recall.", encoding="utf-8")
    second = tmp_path / "second.txt"
    second.write_text("Results\n\nBeta retrieval reduces latency.", encoding="utf-8")
    with CorpusStore(tmp_path / "fts.sqlite") as store:
        if not store.fts_available:
            return
        store.upsert(parse_document(first, paper_id="first"))
        assert store.search("alpha recall", top_k=1)[0].paper.id == "first"
        store.upsert(parse_document(second, paper_id="second"))
        assert store.search("beta latency", top_k=1)[0].paper.id == "second"


def test_retrieve_evidence_filters_before_applying_limit(tmp_path: Path) -> None:
    with CorpusStore(tmp_path / "scoped.sqlite") as store:
        for index in range(8):
            source = tmp_path / f"paper-{index}.txt"
            source.write_text(
                f"Results\n\nGRPO improves reasoning performance number {index}.",
                encoding="utf-8",
            )
            store.upsert(parse_document(source, paper_id=f"paper-{index}"))

        results = retrieve_evidence(store, "paper-7", "GRPO reasoning", top_k=1)

    assert len(results) == 1
    assert results[0].paper.id == "paper-7"


def test_search_finds_latin_acronym_without_spaces_in_chinese_query(tmp_path: Path) -> None:
    source = tmp_path / "grpo.txt"
    source.write_text("Abstract\n\nGRPO improves reasoning performance.", encoding="utf-8")
    with CorpusStore(tmp_path / "mixed-query.sqlite") as store:
        store.upsert(parse_document(source, paper_id="grpo-paper"))
        results = search_papers(store, "查找grpo相关的论文", top_k=3)

    assert results
    assert results[0].paper.id == "grpo-paper"
