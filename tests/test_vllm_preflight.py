import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "preflight_vllm.py"
SPEC = importlib.util.spec_from_file_location("preflight_vllm", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
preflight_vllm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight_vllm)


def _write_model_manifest(model_path: Path) -> None:
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    (model_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_path / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.layer.weight": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )


def test_model_inspection_accepts_complete_manifest(tmp_path: Path) -> None:
    _write_model_manifest(tmp_path)

    details, errors = preflight_vllm.inspect_model_directory(tmp_path)

    assert errors == []
    assert details["shards"] == {"expected": 1, "present": 1}


def test_model_inspection_detects_missing_shard(tmp_path: Path) -> None:
    _write_model_manifest(tmp_path)
    (tmp_path / "model-00001-of-00001.safetensors").unlink()

    _, errors = preflight_vllm.inspect_model_directory(tmp_path)

    assert any("Model shard files" in error for error in errors)


def test_endpoint_probe_rejects_external_urls() -> None:
    details, errors = preflight_vllm.inspect_endpoint("https://example.invalid/v1", 1)

    assert details["url"] == "https://example.invalid/v1"
    assert errors == ["Endpoint probe must use localhost, 127.0.0.1, or ::1."]


def test_endpoint_probe_accepts_loopback_urls() -> None:
    assert preflight_vllm.is_loopback_url("http://127.0.0.1:8001/v1")
    assert preflight_vllm.is_loopback_url("http://[::1]:8001/v1")
    assert not preflight_vllm.is_loopback_url("https://www.micuapi.ai/v1")
    assert not preflight_vllm.is_loopback_url("http://127.0.0.1:8001/v1/responses")
    assert not preflight_vllm.is_loopback_url("http://127.0.0.1:8001/v1/%72esponses")
    assert not preflight_vllm.is_loopback_url("http://127.0.0.1:8001/v1%2Fresponses")
    assert not preflight_vllm.is_loopback_url("http://127.0.0.1:8001/")
    assert not preflight_vllm.is_loopback_url("http://127.0.0.1:8001/v1?redirect=external")


def test_endpoint_probe_does_not_follow_redirects() -> None:
    class RedirectHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(302)
            self.send_header("Location", "/v1/redirect-target")
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        details, errors = preflight_vllm.inspect_endpoint(
            f"http://127.0.0.1:{server.server_port}/v1", 1
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

    assert details["reachable"] is False
    assert errors == ["Local model endpoint returned HTTP 302."]
