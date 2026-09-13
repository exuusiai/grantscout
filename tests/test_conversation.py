from paperscout.agent.conversation import ConversationMessage, understand_request
from paperscout.config import Settings


def test_ambiguous_term_triggers_clarification_on_model_failure(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "paperscout.agent.conversation.OpenAICompatibleClient.chat_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("offline")),
    )

    result = understand_request(
        [ConversationMessage(role="user", content="查找基架相关论文")],
        Settings(data_dir=tmp_path),
    )

    assert result.status == "clarification"
    assert "传统系统基础架构" in result.message
    assert result.refined_question is None


def test_model_can_resolve_follow_up_into_search_task(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "paperscout.agent.conversation.OpenAICompatibleClient.chat_json",
        lambda *args, **kwargs: {
            "status": "ready",
            "message": "已明确为大模型系统基础架构。",
            "refined_question": "大语言模型推理系统基础架构",
            "ranking": "relevance",
        },
    )
    messages = [
        ConversationMessage(role="user", content="查找基架相关论文"),
        ConversationMessage(role="assistant", content="你指哪一种基架？"),
        ConversationMessage(role="user", content="大模型系统基础架构"),
    ]

    result = understand_request(messages, Settings(data_dir=tmp_path))

    assert result.status == "ready"
    assert result.refined_question == "大语言模型推理系统基础架构"
