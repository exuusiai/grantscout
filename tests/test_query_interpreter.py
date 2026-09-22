from grantscout.config import Settings
from grantscout.models.llm import ChatResponse
from grantscout.retrieval.query_interpreter import interpret_query


def test_chinese_query_is_translated_and_cached(tmp_path, monkeypatch) -> None:
    calls = []

    def chat(self, messages, max_tokens, temperature, chat_template_kwargs=None):
        calls.append(messages[-1]["content"])
        return ChatResponse(
            content='{"search_query":"embodied intelligence robot manipulation","ranking":"recent"}',
            model="test",
            usage={},
        )

    monkeypatch.setattr("grantscout.models.llm.OpenAICompatibleClient.chat", chat)
    settings = Settings(data_dir=tmp_path)

    first = interpret_query("查找最新的具身智能与机器人操作论文", settings)
    second = interpret_query("查找最新的具身智能与机器人操作论文", settings)

    assert first == ("embodied intelligence robot manipulation", "recent")
    assert second == first
    assert len(calls) == 1


def test_english_query_does_not_call_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "grantscout.models.llm.OpenAICompatibleClient.chat",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("model called")),
    )

    assert interpret_query("Find GRPO papers", Settings(data_dir=tmp_path)) == ("GRPO", None)
