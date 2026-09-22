import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from grantscout.completion import CompletionCache, CompletionEngine
from grantscout.completion.filters import parse_filters
from grantscout.completion.schemas import CompletionRequest
from grantscout.config import Settings
from grantscout.models.llm import OpenAICompatibleClient
from grantscout.models.schemas import EvidenceItem, Paper, PaperSection, ParsedPaper
from grantscout.retrieval.store import CorpusStore

EMANE_TEXT = (
    "EMANE是由美国波音公司开源的网络仿真平台,其基于容器技术等云原生技术,"
    "专为移动自组织网络和无线通信系统的研究、开发和测试而设计。"
    "它广泛应用于军事通信网络仿真领域。"
)
STARRYNET_TEXT = (
    "StarryNet is a container-based network emulator for LEO satellite constellations "
    "released in 2023, and it supports large scale constellation emulation."
)
LEGACY_TEXT = (
    "Legacy satellite emulators from 2021 rely on discrete event simulation only, "
    "which limits real time interaction for digital twin experiments."
)


class FakeChatModel:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    def chat(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(content=self.content, model="fake", usage={})

    def chat_json(self, **kwargs):  # pragma: no cover - not used by the engine
        raise AssertionError("completion engine must not request JSON")


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        use_model_reasoning=False,
    )


def _build_corpus(tmp_path: Path) -> Path:
    corpus_path = tmp_path / "corpus.sqlite"
    papers = [
        ParsedPaper(
            paper=Paper(
                id="emane",
                title="EMANE Network Emulator Study",
                authors=["Author One"],
                year=2023,
                abstract="",
                source_path="library/emane.pdf",
            ),
            sections=[
                PaperSection(
                    id="emane:section:0000", paper_id="emane", title="Body", section_index=0, text=EMANE_TEXT
                )
            ],
            evidence_items=[
                EvidenceItem(
                    id="emane:section:0000:evidence:0000",
                    paper_id="emane",
                    section_id="emane:section:0000",
                    text=EMANE_TEXT,
                )
            ],
        ),
        ParsedPaper(
            paper=Paper(
                id="starrynet",
                title="StarryNet Emulator Paper",
                authors=["Author Two"],
                year=2023,
                abstract="",
                source_path="library/starrynet.pdf",
            ),
            sections=[
                PaperSection(
                    id="starrynet:section:0000",
                    paper_id="starrynet",
                    title="Body",
                    section_index=0,
                    text=STARRYNET_TEXT,
                )
            ],
            evidence_items=[
                EvidenceItem(
                    id="starrynet:section:0000:evidence:0000",
                    paper_id="starrynet",
                    section_id="starrynet:section:0000",
                    text=STARRYNET_TEXT,
                )
            ],
        ),
        ParsedPaper(
            paper=Paper(
                id="legacy",
                title="Legacy Satellite Emulation 2021",
                authors=["Author Three"],
                year=2021,
                abstract="",
                source_path="library/legacy.pdf",
            ),
            sections=[
                PaperSection(
                    id="legacy:section:0000", paper_id="legacy", title="Body", section_index=0, text=LEGACY_TEXT
                )
            ],
            evidence_items=[
                EvidenceItem(
                    id="legacy:section:0000:evidence:0000",
                    paper_id="legacy",
                    section_id="legacy:section:0000",
                    text=LEGACY_TEXT,
                )
            ],
        ),
    ]
    with CorpusStore(corpus_path) as store:
        for document in papers:
            store.upsert(document)
        store.set_paper_meta("emane", tags=["emulator", "military"], cited_by_count=12, venue="MILCOM")
        store.set_paper_meta("starrynet", tags=["emulator"], cited_by_count=3, venue="NSDI")
        store.set_paper_meta("legacy", tags=["legacy"], cited_by_count=30, venue="IEEE Transactions")
    return corpus_path


def _engine(tmp_path: Path, model=None, cache: CompletionCache | None = None) -> CompletionEngine:
    with CorpusStore(tmp_path / "corpus.sqlite") as store:
        pass
    store = CorpusStore(tmp_path / "corpus.sqlite")
    engine = CompletionEngine(store, _settings(tmp_path), cache=cache)
    if model is not None:
        engine.model_client = model
    return engine


def test_parse_filters_supports_value_keys_and_warns_on_pending_metadata() -> None:
    parsed = parse_filters("前言 @{filename:emane.pdf} @{time:2021-2023} @{bogus:1} 正文")
    assert parsed.clean_text == "前言 正文"
    assert parsed.filters.filename == ["emane.pdf"]
    assert (parsed.filters.time_from, parsed.filters.time_to) == (2021, 2023)
    assert any("bogus" in warning for warning in parsed.warnings)

    recent = parse_filters("@{time:recent3} 文本")
    assert recent.filters.time_from is not None

    impact = parse_filters("@{impact:>5} 文本")
    assert impact.filters.impact_min == 5
    conference = parse_filters("@{conference:CVPR,ICCV} 文本")
    assert conference.filters.conferences == ["CVPR", "ICCV"]

    cleared = parse_filters("@{filename:a} @{author:b} @{unset} 文本")
    assert cleared.filters.is_empty()


def test_tag_impact_conference_filters_apply(tmp_path: Path) -> None:
    _build_corpus(tmp_path)
    engine = _engine(tmp_path)

    tagged = engine.complete(CompletionRequest(prefix="@{tag:emulator} emulator", mode="lexical"))
    papers = {citation.paper_id for c in tagged.candidates for citation in c.citations}
    assert papers <= {"emane", "starrynet"}
    assert tagged.filters_applied.get("tag") == ["emulator"]

    impact = engine.complete(CompletionRequest(prefix="@{impact:>5} satellite", mode="lexical"))
    impact_papers = {citation.paper_id for c in impact.candidates for citation in c.citations}
    assert "starrynet" not in impact_papers

    venue = engine.complete(CompletionRequest(prefix="@{conference:NSDI} emulator", mode="lexical"))
    venue_papers = {citation.paper_id for c in venue.candidates for citation in c.citations}
    assert venue_papers <= {"starrynet"}


def test_scope_labels_and_privacy_gate(tmp_path: Path) -> None:
    _build_corpus(tmp_path)
    with CorpusStore(tmp_path / "corpus.sqlite") as store:
        store.set_paper_meta("emane", scope="private")

    engine = _engine(tmp_path)
    response = engine.complete(CompletionRequest(prefix="根据调研,EMANE是由美国波音", mode="lexical"))
    scopes = {citation.scope for c in response.candidates for citation in c.citations if citation.paper_id == "emane"}
    assert scopes == {"private"}

    engine.model_client = OpenAICompatibleClient(
        base_url="https://api.example.com/v1",
        api_key="key",
        model="remote-model",
        allow_research_endpoint=True,
    )
    blocked = engine.complete(CompletionRequest(prefix="EMANE是由美国波音", mode="llm"))
    assert not [c for c in blocked.candidates if c.kind == "llm"]
    assert any("私域" in warning for warning in blocked.filter_warnings)


def test_private_upload_flows_into_completion_scope(tmp_path: Path, monkeypatch) -> None:
    import time as time_module

    from grantscout.api import app as app_module
    from grantscout.knowledge import KnowledgeService
    from grantscout.retrieval.store import CorpusStore as Store

    settings = _settings(tmp_path)
    monkeypatch.setattr(app_module, "get_settings", lambda: settings)
    app_module.app.state.knowledge = KnowledgeService(settings.data_dir / "knowledge")
    client = TestClient(app_module.app)

    project = client.post("/api/projects", json={"name": "Lab Private"}).json()
    uploaded = client.post(
        "/api/knowledge/upload",
        data={"project_id": project["id"], "scope": "private"},
        files=[("files", ("lab-notes.md", "# 实验记录\n\n星地链路切换时延在实验室环境中实测为 120 毫秒。\n".encode("utf-8"), "text/markdown"))],
    )
    assert uploaded.status_code == 200
    document_id = uploaded.json()[0]["id"]
    status = ""
    for _ in range(300):
        status = client.get(f"/api/knowledge/tasks/{document_id}").json()["status"]
        if status in {"ready", "failed"}:
            break
        time_module.sleep(0.1)
    assert status == "ready", f"document ingestion ended as {status!r}"

    knowledge = app_module.app.state.knowledge
    corpus_path = knowledge.corpus_path(project["id"])
    with Store(corpus_path) as store:
        meta = store.list_paper_meta()
        assert meta[document_id]["scope"] == "private"

    completion = client.post(
        "/api/completion",
        json={
            "prefix": "根据实验记录,星地链路切换时延在实验室环境中实测为",
            "corpus": str(corpus_path),
            "mode": "lexical",
        },
    )
    assert completion.status_code == 200, completion.text
    payload = completion.json()
    assert payload["candidates"]
    scopes = {citation["scope"] for c in payload["candidates"] for citation in c["citations"]}
    assert "private" in scopes

    meta_updated = client.patch(
        "/api/corpus/paper-meta",
        json={"corpus": str(corpus_path), "paper_id": document_id, "venue": "内部资料"},
    )
    assert meta_updated.status_code == 200
    assert meta_updated.json()["venue"] == "内部资料"


def test_verbatim_completion_returns_library_continuation(tmp_path: Path) -> None:
    _build_corpus(tmp_path)
    engine = _engine(tmp_path)
    response = engine.complete(CompletionRequest(prefix="根据调研,EMANE是由美国波音", max_candidates=3))

    assert response.candidates, "verbatim path must find the library sentence"
    top = response.candidates[0]
    assert top.kind == "verbatim"
    assert top.text.startswith("公司开源的网络仿真平台")
    assert top.citations and top.citations[0].paper_id == "emane"
    assert response.latency_ms >= 0


def test_lexical_mode_respects_time_filter(tmp_path: Path) -> None:
    _build_corpus(tmp_path)
    engine = _engine(tmp_path)
    scoped = engine.complete(
        CompletionRequest(prefix="@{time:2021-2021} legacy satellite", mode="lexical")
    )
    assert scoped.candidates
    assert all(citation.paper_id == "legacy" for candidate in scoped.candidates for citation in candidate.citations)
    assert scoped.filters_applied.get("time") == ["2021-2021"]

    filtered_out = engine.complete(
        CompletionRequest(prefix="@{filename:starrynet.pdf} legacy satellite", mode="lexical")
    )
    papers = {citation.paper_id for candidate in filtered_out.candidates for citation in candidate.citations}
    assert papers <= {"starrynet"}


def test_llm_candidate_parses_citations_and_cache_hits(tmp_path: Path) -> None:
    _build_corpus(tmp_path)
    fake = FakeChatModel("该平台已被广泛验证【文献1】,并支持大规模组网仿真。")
    cache = CompletionCache()
    engine = _engine(tmp_path, model=fake, cache=cache)
    request = CompletionRequest(prefix="EMANE是由美国波音", mode="llm")

    first = engine.complete(request)
    assert first.candidates and first.candidates[0].kind == "llm"
    assert first.candidates[0].citations[0].evidence_id == "emane:section:0000:evidence:0000"
    assert first.cache_hit is False

    second = engine.complete(request)
    assert second.cache_hit is True
    assert second.candidates[0].text == first.candidates[0].text

    stale = CompletionRequest(prefix="EMANE是由美国波音", mode="llm", session_id="s1", request_id="r1")
    cache.register("s1", "r1")
    cache.register("s1", "r2")
    response = engine.complete(stale)
    assert response.superseded is True


def test_commands_summarize_refine_find(tmp_path: Path) -> None:
    _build_corpus(tmp_path)
    fake = FakeChatModel("小结:云原生仿真平台已成为卫星网络研究的主流底座。")
    engine = _engine(tmp_path, model=fake)

    refined = engine.run_command(
        CompletionRequest(command="refine", command_text="这个平台很好用,性能很高。")
    )
    assert refined.candidates[0].trigger == "/refine"
    assert refined.candidates[0].text

    engine_fallback = _engine(tmp_path)
    summary = engine_fallback.run_command(
        CompletionRequest(command="summarize", command_text=EMANE_TEXT)
    )
    assert summary.candidates, "deterministic extractive summary must exist"
    assert any("模型未启用" in warning for warning in summary.filter_warnings)

    found = engine.run_command(CompletionRequest(command="find", command_text="satellite emulator"))
    assert found.candidates
    assert {candidate.kind for candidate in found.candidates} <= {"lexical", "semantic", "verbatim"}


def test_api_completion_endpoint(tmp_path: Path, monkeypatch) -> None:
    from grantscout.api import app as app_module

    _build_corpus(tmp_path / "data")
    settings = _settings(tmp_path)
    monkeypatch.setattr(app_module, "get_settings", lambda: settings)
    client = TestClient(app_module.app)

    response = client.post(
        "/api/completion",
        json={"prefix": "根据调研,EMANE是由美国波音", "max_candidates": 3},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["candidates"]
    kinds = {candidate["kind"] for candidate in payload["candidates"]}
    assert "verbatim" in kinds

    outside = client.post(
        "/api/completion",
        json={"prefix": "abc", "corpus": str(tmp_path / "outside.sqlite")},
    )
    assert outside.status_code == 422

    command = client.post(
        "/api/completion",
        json={"command": "find", "command_text": "satellite emulator"},
    )
    assert command.status_code == 200
    assert command.json()["candidates"]

def test_viewer_and_citation_source_url(tmp_path: Path, monkeypatch) -> None:
    from grantscout.api import app as app_module

    _build_corpus(tmp_path)
    _build_corpus(tmp_path / "data")
    (tmp_path / "data" / "library").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "library" / "emane.pdf").write_bytes(b"%PDF-1.4 test")
    settings = _settings(tmp_path)
    monkeypatch.setattr(app_module, "get_settings", lambda: settings)
    client = TestClient(app_module.app)

    engine = _engine(tmp_path)
    response = engine.complete(CompletionRequest(prefix="根据调研,EMANE是由美国波音", mode="lexical"))
    citation = response.candidates[0].citations[0]
    assert citation.source_path and citation.source_url
    assert citation.source_url == "/viewer?file=library/emane.pdf"

    page = client.get(f"/viewer?file={citation.source_path}&page=2")
    assert page.status_code == 200
    assert "#page=2" in page.text and "/files/" in page.text

    served = client.get("/files/library/emane.pdf")
    assert served.status_code == 200

    escape = client.get("/viewer?file=../.env")
    assert escape.status_code == 422
    missing = client.get("/files/does/not/exist.pdf")
    assert missing.status_code == 404
