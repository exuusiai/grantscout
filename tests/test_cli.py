import json

from typer.testing import CliRunner

from grantscout.cli import app


def test_ask_dry_run_returns_structured_json() -> None:
    result = CliRunner().invoke(app, ["ask", "How do RAG systems reduce hallucinations?", "--dry-run"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "planned"
    assert payload["trajectory"][0]["action"] == "receive_question"
