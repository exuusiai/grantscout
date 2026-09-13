import json

from fastapi.testclient import TestClient

from paperscout.api.app import app
from paperscout.config import Settings
from paperscout.retrieval.parser import parse_document
from paperscout.retrieval.store import CorpusStore
from paperscout.retrieval.arxiv import ArxivSearchError
from paperscout.agent.conversation import ConversationResult


def test_api_health_has_corpus_status() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_endpoint_returns_clarification(monkeypatch) -> None:
    monkeypatch.setattr(
        "paperscout.api.app.understand_request",
        lambda messages, settings, locale: ConversationResult(
            status="clarification", message="你指的是哪一种基架？"
        ),
    )

    response = TestClient(app).post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "查找基架相关论文"}], "locale": "zh"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "clarification"


def test_knowledge_api_uploads_document(tmp_path, monkeypatch) -> None:
    from paperscout.knowledge import KnowledgeService

    app.state.knowledge = KnowledgeService(tmp_path)
    client = TestClient(app)
    project = client.post("/api/projects", json={"name": "Guide"}).json()
    response = client.post(
        "/api/knowledge/upload",
        data={"project_id": project["id"]},
        files=[("files", ("guide.md", b"# Guide\n\nProject-specific instructions.", "text/markdown"))],
    )

    assert response.status_code == 200
    assert response.json()[0]["project_id"] == project["id"]


def test_project_conversation_is_persisted_and_deletable(tmp_path, monkeypatch) -> None:
    from paperscout.knowledge import KnowledgeService

    app.state.knowledge = KnowledgeService(tmp_path)
    monkeypatch.setattr(
        "paperscout.api.app.understand_request",
        lambda messages, settings, locale: ConversationResult(
            status="clarification", message="Which scope?"
        ),
    )
    client = TestClient(app)
    project = client.post("/api/projects", json={"name": "Agent workspace"}).json()
    result = client.post(
        "/api/chat",
        json={
            "project_id": project["id"],
            "messages": [{"role": "user", "content": "Find infrastructure papers"}],
            "locale": "en",
        },
    ).json()

    conversation = client.get(f"/api/conversations/{result['conversation_id']}").json()
    assert [message["role"] for message in conversation["messages"]] == ["user", "assistant"]
    assert client.delete(f"/api/conversations/{result['conversation_id']}").status_code == 204
    assert client.delete(f"/api/projects/{project['id']}").status_code == 204


def test_web_ui_is_chinese_first_and_has_visible_run_feedback() -> None:
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert '<html lang="zh-CN">' in response.text
    assert "开始综述" in response.text
    assert "PaperScout 论文侦察" in response.text
    assert "就绪 / Ready" not in response.text
    assert "运行进度 / Timeline" not in response.text
    assert 'value="arxiv" selected' in response.text
    assert "正在分析，请稍候" in response.text
    assert "buffer.split('\\n')" in response.text
    assert "buffer.split('\n')" not in response.text
    assert "finally{run.disabled=!resolvedQuestion" in response.text
    assert '<article id="report"' in response.text
    assert "reportFragments?.[locale]||event.report_fragment" in response.text
    assert "locale,ranking:ranking.value})" in response.text
    assert "reportFragments[locale]" in response.text
    assert "report-scroll{height:645px;overflow-y:auto" in response.text
    assert 'class="side-stack"' in response.text
    assert "toolCopy=" in response.text
    assert 'type="search"' in response.text
    assert 'id="ranking"' in response.text
    assert "ranking:ranking.value" in response.text
    assert 'id="recommendations"' in response.text
    assert "latestPapers.slice(0,3)" in response.text
    assert "replace('/abs/','/pdf/')" in response.text
    assert "window.print()" in response.text
    assert 'id="chat-log"' in response.text
    assert 'id="send"' in response.text
    assert "jsonRequest('/api/chat'" in response.text
    assert "type=\"button\" disabled>重新生成" in response.text
    assert "setTimeout(runReview,0)" in response.text
    assert "status.textContent=t('autoStarting')" in response.text
    assert 'id="source-project"' in response.text
    assert "project_id:project.value||null" in response.text
    assert 'id="conversation"' in response.text
    assert "deleteCurrentProject" in response.text
    assert "archived_report" in response.text


def test_arxiv_failure_is_visible_instead_of_returning_local_results(monkeypatch) -> None:
    monkeypatch.setattr(
        "paperscout.api.app.search_arxiv",
        lambda *args, **kwargs: (_ for _ in ()).throw(ArxivSearchError("rate limited")),
    )

    response = TestClient(app).post("/api/ask", json={"question": "Find GRPO papers"})

    assert response.status_code == 503
    assert "避免混入无关论文" in response.json()["detail"]


def test_api_streams_tool_events_and_final_report(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source = tmp_path / "paper.txt"
    source.write_text("Results\n\nThe method improves evidence recall.", encoding="utf-8")
    corpus = data_dir / "corpus.sqlite"
    with CorpusStore(corpus) as store:
        store.upsert(parse_document(source, paper_id="stream-paper"))
    monkeypatch.setattr(
        "paperscout.api.app.get_settings",
        lambda: Settings(data_dir=data_dir, runs_dir=tmp_path / "runs"),
    )

    response = TestClient(app).post(
        "/api/ask/stream",
        json={"question": "What improves evidence recall?", "source": "local"},
    )
    events = [json.loads(line) for line in response.text.splitlines()]

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert any(event["type"] == "tool_call" for event in events)
    assert events[-1]["type"] == "completed"
    assert "## Citation Audit" not in events[-1]["report"]
    assert "## Evidence Table" not in events[-1]["report"]
    assert "## Run Metadata" not in events[-1]["report"]
    assert "<h1>PaperScout 研究笔记" in events[-1]["report_fragment"]
    assert "<article" not in events[-1]["report_fragment"]
    assert set(events[-1]["report_fragments"]) == {"zh", "en"}
