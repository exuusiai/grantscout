from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from grantscout.config import Settings
from grantscout.models.llm import ModelClientError, OpenAICompatibleClient
from grantscout.retrieval.arxiv import build_arxiv_query


_CHINESE = re.compile(r"[\u3400-\u9fff]")
_RANKINGS = {"relevance", "recent", "citations"}


def interpret_query(question: str, settings: Settings) -> tuple[str, str | None]:
    """Translate Chinese research intent into English terms using only the local model."""
    fallback = build_arxiv_query(question)
    if not _CHINESE.search(question):
        return fallback, None

    cache_dir = Path(settings.data_dir) / "query-cache"
    key = hashlib.sha256(f"v1\0{question.strip().casefold()}".encode()).hexdigest()
    cache_path = cache_dir / f"{key}.json"
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return _validated(cached, fallback)
    except (OSError, ValueError, TypeError):
        pass

    client = OpenAICompatibleClient(
        settings.model_base_url,
        settings.model_api_key,
        settings.model_name,
        timeout_seconds=min(settings.model_timeout_seconds, 30.0),
    )
    try:
        result = client.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "Convert every Chinese concept in the user's literature-search request "
                        "to concise English academic search keywords. Preserve acronyms. Infer "
                        "ranking only when explicitly requested. Return JSON only: "
                        '{"search_query":"...","ranking":"relevance|recent|citations"}.'
                    ),
                },
                {"role": "user", "content": question},
            ],
            max_tokens=160,
            temperature=0.0,
        )
        interpreted = _validated(result, fallback)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {"search_query": interpreted[0], "ranking": interpreted[1]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return interpreted
    except (ModelClientError, OSError, ValueError, TypeError):
        return fallback, None


def _validated(value: object, fallback: str) -> tuple[str, str | None]:
    if not isinstance(value, dict):
        return fallback, None
    query = value.get("search_query")
    if not isinstance(query, str) or not query.strip() or _CHINESE.search(query):
        query = fallback
    ranking = value.get("ranking")
    return query.strip(), ranking if ranking in _RANKINGS else None
