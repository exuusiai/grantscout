from urllib.parse import unquote, urlsplit


LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
LOCAL_API_BASE_PATH = "/v1"


def normalize_local_base_url(value: str) -> str:
    """Validate and normalize the local OpenAI-compatible API base URL."""
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    decoded_path = unquote(parsed.path).rstrip("/")

    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in LOOPBACK_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or decoded_path != LOCAL_API_BASE_PATH
    ):
        raise ValueError(
            "model_base_url must point to a local inference server at the exact "
            "local inference endpoint /v1 on loopback; third-party gateways, "
            "ambiguous paths, and /responses endpoints are not permitted"
        )
    return normalized


def normalize_research_base_url(value: str) -> str:
    """Validate an explicitly configured HTTPS OpenAI-compatible research endpoint."""
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    decoded_path = unquote(parsed.path).rstrip("/")
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or decoded_path != LOCAL_API_BASE_PATH:
        raise ValueError("research_model_base_url must be an HTTPS OpenAI-compatible endpoint ending in /v1")
    return normalized
