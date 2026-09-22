from grantscout.models.llm import OpenAICompatibleClient, parse_json_content


def test_parse_json_content_ignores_thinking_and_code_fences() -> None:
    payload = parse_json_content(
        '<think>reasoning is not part of the contract</think>\n```json\n{"ok": true}\n```'
    )

    assert payload == {"ok": True}


def test_client_strips_trailing_slash() -> None:
    client = OpenAICompatibleClient(
        base_url="http://localhost:8000/v1/",
        api_key="EMPTY",
        model="demo",
    )

    assert client.base_url == "http://localhost:8000/v1"


def test_client_rejects_external_endpoint() -> None:
    try:
        OpenAICompatibleClient(
            base_url="https://example.invalid/v1",
            api_key="EMPTY",
            model="demo",
        )
    except ValueError as error:
        assert "local inference endpoint" in str(error)
    else:
        raise AssertionError("Client accepted an external model endpoint")


def test_client_rejects_gateway_and_responses_endpoint() -> None:
    for base_url in (
        "https://www.micuapi.ai/v1",
        "http://127.0.0.1:8001/v1/responses",
        "http://127.0.0.1:8001/v1/%72esponses",
        "http://127.0.0.1:8001/v1/../responses",
        "http://127.0.0.1:8001/v1%2Fresponses",
        "http://127.0.0.1:8001/",
        "http://127.0.0.1:8001/v1?redirect=external",
    ):
        try:
            OpenAICompatibleClient(base_url=base_url, api_key="EMPTY", model="demo")
        except ValueError as error:
            assert "loopback" in str(error)
        else:
            raise AssertionError(f"Client accepted unsafe endpoint: {base_url}")
