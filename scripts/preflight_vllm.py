#!/usr/bin/env python3
"""Validate a local Qwen/vLLM deployment before starting the model server.

This script deliberately accepts only a loopback endpoint for the optional
post-start probe. GrantScout's model client has the same restriction, so a
third-party OpenAI-compatible gateway cannot accidentally enter this workflow.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
LOCAL_API_BASE_PATH = "/v1"
REQUIRED_MODEL_FILES = ("config.json", "tokenizer.json", "model.safetensors.index.json")


class NoRedirectHandler(HTTPRedirectHandler):
    """Treat redirects from the local model service as failed probes."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def is_loopback_url(url: str) -> bool:
    """Return whether ``url`` is a safe loopback API base URL."""
    parsed = urlsplit(url)
    decoded_path = unquote(parsed.path).rstrip("/")
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname in LOOPBACK_HOSTS
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
        and decoded_path == LOCAL_API_BASE_PATH
    )


def inspect_model_directory(model_path: Path) -> tuple[dict[str, Any], list[str]]:
    """Inspect the Hugging Face model manifest without loading model weights."""
    details: dict[str, Any] = {"path": str(model_path), "files": {}}
    errors: list[str] = []

    if not model_path.is_dir():
        return details, [f"Model directory is unavailable: {model_path}"]

    for name in REQUIRED_MODEL_FILES:
        path = model_path / name
        exists = path.is_file() and path.stat().st_size > 0
        details["files"][name] = exists
        if not exists:
            errors.append(f"Required model file is missing or empty: {path}")

    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file() or index_path.stat().st_size == 0:
        return details, errors

    try:
        manifest = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = manifest.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("weight_map is missing or empty")
        shard_names = sorted(set(weight_map.values()))
        if not all(isinstance(name, str) for name in shard_names):
            raise ValueError("weight_map contains a non-string shard name")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"Cannot read model shard manifest: {error}")
        return details, errors

    missing_shards = [
        name
        for name in shard_names
        if not (model_path / name).is_file() or (model_path / name).stat().st_size == 0
    ]
    details["shards"] = {
        "expected": len(shard_names),
        "present": len(shard_names) - len(missing_shards),
    }
    if missing_shards:
        errors.append(f"Model shard files are missing or empty: {', '.join(missing_shards[:5])}")
    return details, errors


def inspect_runtime() -> tuple[dict[str, Any], list[str]]:
    """Check that vLLM and a CUDA-capable Torch build can serve this model."""
    details: dict[str, Any] = {}
    errors: list[str] = []

    try:
        details["vllm_version"] = importlib.metadata.version("vllm")
        import vllm  # noqa: F401
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        errors.append(f"vLLM is unavailable in {sys.executable}: {error}")

    try:
        import torch
    except ImportError as error:
        return details, [*errors, f"PyTorch is unavailable in {sys.executable}: {error}"]

    details["torch_version"] = torch.__version__
    cuda_available = torch.cuda.is_available()
    details["cuda_available"] = cuda_available
    if not cuda_available:
        return details, [
            *errors,
            "PyTorch cannot access a CUDA GPU; vLLM requires a compatible NVIDIA runtime.",
        ]

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    target_arch = f"sm_{capability[0]}{capability[1]}"
    supported_arches = torch.cuda.get_arch_list()
    details["gpu"] = {
        "name": device_name,
        "compute_capability": target_arch,
        "torch_arches": supported_arches,
    }
    if supported_arches and not arch_compatible(capability, supported_arches):
        errors.append(
            f"Installed PyTorch does not include kernels compatible with {target_arch} ({device_name}). "
            "Install a PyTorch build compatible with this GPU before starting vLLM."
        )
    return details, errors


def arch_compatible(capability: tuple[int, int], supported_arches: list[str]) -> bool:
    """Same-major lower-minor kernels run on the GPU (CUDA binary compatibility).

    e.g. sm_80/sm_86 kernels run on an sm_89 GPU (L20); requiring the exact
    sm_89 string in torch's arch list produced false negatives on common GPUs.
    """
    gpu_major, gpu_minor = capability
    for arch in supported_arches:
        match = re.fullmatch(r"sm_(\d+)(a)?", arch)
        if not match:
            continue
        digits = match.group(1)
        major = int(digits) // 10
        minor = int(digits) % 10
        if major == gpu_major and minor <= gpu_minor:
            return True
    return False


def inspect_endpoint(
    endpoint: str, timeout_seconds: float
) -> tuple[dict[str, Any], list[str]]:
    """Probe the local vLLM model listing after the server has started."""
    if not is_loopback_url(endpoint):
        return {"url": endpoint}, ["Endpoint probe must use localhost, 127.0.0.1, or ::1."]

    url = endpoint.rstrip("/") + "/models"
    try:
        request = Request(url, headers={"Accept": "application/json"})
        opener = build_opener(NoRedirectHandler())
        with opener.open(request, timeout=timeout_seconds) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
        models = payload.get("data", []) if isinstance(payload, dict) else []
        return {"url": url, "reachable": True, "models": models}, []
    except HTTPError as error:
        return {"url": url, "reachable": False, "http_status": error.code}, [
            f"Local model endpoint returned HTTP {error.code}."
        ]
    except (OSError, URLError, ValueError, json.JSONDecodeError) as error:
        return {"url": url, "reachable": False}, [
            f"Local model endpoint is unavailable: {error}"
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Local Hugging Face model directory.",
    )
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="Validate only model files; do not require vLLM, PyTorch, or CUDA.",
    )
    parser.add_argument(
        "--check-endpoint",
        metavar="URL",
        help="Also request GET <URL>/models. URL must be a loopback endpoint.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Endpoint probe timeout in seconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0:
        print(
            json.dumps(
                {"status": "failed", "errors": ["--timeout must be greater than zero."]},
                indent=2,
            )
        )
        return 2
    if args.model_only and args.check_endpoint:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "errors": ["--model-only cannot be combined with --check-endpoint."],
                },
                indent=2,
            )
        )
        return 2

    model_details, errors = inspect_model_directory(args.model_path)
    report: dict[str, Any] = {"python": sys.executable, "model": model_details}
    if not args.model_only:
        runtime_details, runtime_errors = inspect_runtime()
        report["runtime"] = runtime_details
        errors.extend(runtime_errors)
    if args.check_endpoint:
        endpoint_details, endpoint_errors = inspect_endpoint(args.check_endpoint, args.timeout)
        report["endpoint"] = endpoint_details
        errors.extend(endpoint_errors)

    report["status"] = "ready" if not errors else "failed"
    report["errors"] = errors
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
