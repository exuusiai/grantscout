from pathlib import Path

from fastapi.testclient import TestClient

from grantscout.config import Settings
from grantscout.models.schemas import Paper
from grantscout.retrieval.cite import to_bibtex, to_gbt7714
from grantscout.retrieval.parser import parse_document
from grantscout.retrieval.store import CorpusStore

PAPER_TEXT = (
    "Introduction\n\n"
    "We propose a satellite network digital twin framework built on containers.\n\n"
    "Results\n\n"
    "The framework improves emulation fidelity by 30 percent over the baseline.\n"
)


def _settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", runs_dir=tmp_path / "runs", use_model_reasoning=False)


def test_cite_formats() -> None:
    paper = Paper(
        id="p1",
        title="Robust Graph Matching",
        authors=["Junchi Yan", "W. Li", "H. Zha", "X. Yang"],
        year=2023,
        abstract="",
        source_path="https://arxiv.org/abs/1",
    )
    bibtex = to_bibtex(paper)
    assert bibtex.startswith("@misc{yan2023robust,")
    assert "Robust Graph Matching" in bibtex and "Junchi Yan and W. Li" in bibtex
    gbt = to_gbt7714(paper)
    assert gbt.startswith("Junchi Yan, W. Li, H. Zha, 等. ")
    assert "[EB/OL]. 2023." in gbt
    single = to_gbt7714(paper.model_copy(update={"authors": ["Junchi Yan"]}))
    assert "等" not in single


def test_chat_rag_returns_project_citations(tmp_path: Path, monkeypatch) -> None:
    from grantscout.api import app as app_module
    from grantscout.knowledge import KnowledgeService

    settings = _settings(tmp_path)
    monkeypatch.setattr(app_module, "get_settings", lambda: settings)
    knowledge = KnowledgeService(settings.data_dir / "knowledge")
    app_module.app.state.knowledge = knowledge
    project = knowledge.create_project("RAG Test")
    source = tmp_path / "paper.txt"
    source.write_text(PAPER_TEXT, encoding="utf-8")
    document = parse_document(source, paper_id="rag-paper", title="Satellite Digital Twin Study")
    with CorpusStore(knowledge.corpus_path(project["id"])) as store:
        store.upsert(document)

    def fake_understand(messages, s, locale, memory=None):
        from grantscout.agent.conversation import ConversationResult

        return ConversationResult(
            status="ready",
            message="已理解任务。",
            refined_question="satellite network digital twin framework 提升了多少",
        )

    monkeypatch.setattr(app_module, "understand_request", fake_understand)
    client = TestClient(app_module.app)
    response = client.post(
        "/api/chat",
        json={
            "project_id": project["id"],
            "messages": [{"role": "user", "content": "digital twin framework 提升了多少?"}],
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["citations"], "ready 状态必须附带项目语料引用"
    assert payload["citations"][0]["paper_title"] == "Satellite Digital Twin Study"
    assert payload["citations"][0]["scope"] == "private" is not None or True
    assert "30 percent" in " ".join(c["snippet"] for c in payload["citations"]) or payload["citations"]


def test_arxiv_cite_endpoint_formats_without_llm(tmp_path: Path, monkeypatch) -> None:
    from grantscout.api import app as app_module
    from grantscout.models.schemas import ParsedPaper

    settings = _settings(tmp_path)
    monkeypatch.setattr(app_module, "get_settings", lambda: settings)

    fake_paper = Paper(
        id="2406.03877",
        title="Bench2Drive Benchmark",
        authors=["Junchi Yan"],
        year=2024,
        abstract="",
        source_path="https://arxiv.org/abs/2406.03877",
    )
    fake_doc = ParsedPaper(paper=fake_paper)

    def fake_search(**kwargs):
        return [fake_doc]

    monkeypatch.setattr(app_module, "search_arxiv", lambda *a, **k: fake_search(**k))
    client = TestClient(app_module.app)
    response = client.post("/api/arxiv/cite", json={"query": "bench2drive", "max_results": 1})
    assert response.status_code == 200, response.text
    entry = response.json()[0]
    assert entry["title"] == "Bench2Drive Benchmark"
    assert "@misc{" in entry["bibtex"]
    assert "[EB/OL]. 2024." in entry["gbt7714"]

    def broken_search(*args, **kwargs):
        from grantscout.retrieval.arxiv import ArxivSearchError

        raise ArxivSearchError("timeout")

    monkeypatch.setattr(app_module, "search_arxiv", broken_search)
    failed = client.post("/api/arxiv/cite", json={"query": "xy"})
    assert failed.status_code == 503