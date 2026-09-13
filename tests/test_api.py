import json

from fastapi.testclient import TestClient

from paperscout.api.app import app
from paperscout.config import Settings
from paperscout.retrieval.parser import parse_document
from paperscout.retrieval.store import CorpusStore


def test_api_health_has_corpus_status() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_web_ui_is_chinese_first_and_has_visible_run_feedback() -> None:
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert '<html lang="zh-CN">' in response.text
    assert "开始综述" in response.text
    assert "PaperScout 论文侦察" in response.text
    assert "正在分析，请稍候" in response.text
    assert "finally{run.disabled=false" in response.text


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

    response = TestClient(app).post("/api/ask/stream", json={"question": "What improves evidence recall?"})
    events = [json.loads(line) for line in response.text.splitlines()]

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert any(event["type"] == "tool_call" for event in events)
    assert events[-1]["type"] == "completed"
    assert "## Citation Audit" in events[-1]["report"]
