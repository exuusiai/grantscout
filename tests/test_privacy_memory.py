import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from grantscout.config import Settings
from grantscout.knowledge import KnowledgeService
from grantscout.privacy.pii import scrub_parsed_paper, scrub_text
from grantscout.models.schemas import EvidenceItem, Paper, PaperSection, ParsedPaper

PII_TEXT = (
    "联系人张三,手机 13812345678,邮箱 zhangsan@lab.edu.cn,身份证 11010119900307889X。"
    "本实验室专注星地链路仿真。"
)


def test_pii_rule_engine() -> None:
    scrubbed = scrub_text(PII_TEXT, extra_words=["星地链路仿真"])
    assert "13812345678" not in scrubbed and "<PHONE>" in scrubbed
    assert "11010119900307889X" not in scrubbed and "<ID_CARD>" in scrubbed
    assert "zhangsan@lab.edu.cn" not in scrubbed and "<EMAIL>" in scrubbed
    assert "星地链路仿真" not in scrubbed and "<CUSTOM>" in scrubbed
    assert "张三" in scrubbed  # NER 层才处理人名,规则层保留
    assert "本实验室专注" in scrubbed  # 非敏感文本保留


def test_pii_scrub_parsed_paper() -> None:
    document = ParsedPaper(
        paper=Paper(id="p1", title="内部报告", abstract="联系 13812345678"),
        sections=[
            PaperSection(id="p1:s0", paper_id="p1", title="正文", section_index=0, text="邮箱 a@b.com")
        ],
        evidence_items=[
            EvidenceItem(id="p1:s0:e0", paper_id="p1", section_id="p1:s0", text="身份证 11010119900307889X")
        ],
    )
    scrubbed = scrub_parsed_paper(document, use_presidio=False)
    assert "13812345678" not in scrubbed.paper.abstract
    assert "a@b.com" not in scrubbed.sections[0].text
    assert "<ID_CARD>" in scrubbed.evidence_items[0].text


def test_private_ingest_scrubs_and_public_keeps(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(data_dir=tmp_path / "data", runs_dir=tmp_path / "runs", pii_presidio=False)
    monkeypatch.setattr("grantscout.config.get_settings", lambda: settings)
    knowledge = KnowledgeService(tmp_path / "data" / "knowledge")
    project = knowledge.create_project("PII Test")
    private_doc = knowledge.enqueue(
        project["id"],
        "private-notes.md",
        "实验联系人:李四,电话 13998887777。\n".encode("utf-8"),
        scope="private",
    )
    public_doc = knowledge.enqueue(
        project["id"],
        "public-notes.md",
        "公开材料:联系电话 13998887777。\n".encode("utf-8"),
        scope="public",
    )
    for doc in (private_doc, public_doc):
        for _ in range(80):
            if knowledge.document(doc["id"])["status"] == "ready":
                break
            time.sleep(0.1)
    from grantscout.retrieval.store import CorpusStore

    with CorpusStore(knowledge.corpus_path(project["id"])) as store:
        texts = {row["paper_id"]: row["text"] for row in store.connection.execute(
            "SELECT paper_id, text FROM evidence"
        ).fetchall()}
    assert "<PHONE>" in texts[private_doc["id"]], "私域文档必须脱敏后入库"
    assert "13998887777" in texts[public_doc["id"]], "公域文档不脱敏"


def test_chat_memory_crud_and_injection(tmp_path: Path, monkeypatch) -> None:
    from grantscout.api import app as app_module

    settings = Settings(
        data_dir=tmp_path / "data", runs_dir=tmp_path / "runs", use_model_reasoning=True
    )
    monkeypatch.setattr(app_module, "get_settings", lambda: settings)
    captured = {}

    def fake_understand(messages, s, locale, memory=None):
        captured["memory"] = memory
        from grantscout.agent.conversation import ConversationResult

        return ConversationResult(status="ready", message="ok", refined_question=messages[-1].content)

    monkeypatch.setattr(app_module, "understand_request", fake_understand)
    monkeypatch.setattr(app_module, "remember_conversation", lambda messages, s: ["偏好中文报告"])
    client = TestClient(app_module.app)

    project = client.post("/api/projects", json={"name": "Memory Test"}).json()
    client.post(
        "/api/chat",
        json={"project_id": project["id"], "messages": [{"role": "user", "content": "我们做卫星仿真"}]},
    )
    memories = client.get(f"/api/projects/{project['id']}/memory").json()
    assert ["偏好中文报告"] == [m["content"] for m in memories]

    again = client.post(
        "/api/chat",
        json={"project_id": project["id"], "messages": [{"role": "user", "content": "继续"}]},
    )
    assert again.status_code == 200
    assert captured["memory"] == ["偏好中文报告"], "第二轮 chat 必须注入记忆"

    client.delete(f"/api/projects/{project['id']}/memory/{memories[0]['id']}")
    assert client.get(f"/api/projects/{project['id']}/memory").json() == []
    assert client.delete(f"/api/projects/{project['id']}/memory/9999").status_code == 404


def test_memory_extraction_requires_model(tmp_path: Path) -> None:
    from grantscout.agent.conversation import ConversationMessage, remember_conversation

    settings = Settings(data_dir=tmp_path / "data", runs_dir=tmp_path / "runs", use_model_reasoning=False)
    messages = [ConversationMessage(role="user", content="hi")]
    assert remember_conversation(messages, settings) == []
