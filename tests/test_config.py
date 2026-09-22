from pathlib import Path

from grantscout.config import Settings


def test_settings_normalize_base_url_and_create_directories(tmp_path: Path) -> None:
    settings = Settings(model_base_url="http://localhost:8000/v1/")

    settings.prepare_directories(tmp_path)

    assert settings.model_base_url == "http://localhost:8000/v1"
    assert (tmp_path / "data").is_dir()
    assert (tmp_path / "runs").is_dir()


def test_settings_reject_invalid_budget() -> None:
    try:
        Settings(max_steps=0)
    except ValueError as error:
        assert "max_steps" in str(error)
    else:
        raise AssertionError("Settings accepted an invalid max_steps value")


def test_settings_rejects_external_model_endpoint() -> None:
    try:
        Settings(model_base_url="https://example.invalid/v1")
    except ValueError as error:
        assert "local inference server" in str(error)
    else:
        raise AssertionError("Settings accepted an external model endpoint")


def test_settings_rejects_responses_path() -> None:
    try:
        Settings(model_base_url="http://127.0.0.1:8001/v1/responses")
    except ValueError as error:
        assert "responses" in str(error)
    else:
        raise AssertionError("Settings accepted a /responses endpoint")
