from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError

from paperscout.models.schemas import Paper, ParsedPaper
from paperscout.retrieval.parser import build_document


class ArxivSearchError(RuntimeError):
    """Raised when the official arXiv API cannot be queried or parsed."""


def build_arxiv_query(question: str) -> str:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+._-]{1,}", question)
    ignored = {"about", "article", "articles", "find", "paper", "papers", "please", "related", "search"}
    selected = [token for token in tokens if token.lower() not in ignored]
    return " ".join(dict.fromkeys(selected)) or question.strip()


def search_arxiv(
    question: str,
    max_results: int = 20,
    timeout_seconds: float = 12.0,
    api_url: str = "https://export.arxiv.org/api/query",
    cache_dir: Path | None = None,
    max_retries: int = 2,
) -> list[ParsedPaper]:
    query = build_arxiv_query(question)
    cache_path = _cache_path(cache_dir, query, max_results) if cache_dir else None
    parameters = urllib.parse.urlencode(
        {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    request = urllib.request.Request(
        f"{api_url}?{parameters}",
        headers={"User-Agent": "PaperScout/0.1 (research literature search)"},
    )
    last_error: Exception | None = None
    root = None
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                payload = response.read()
            root = ET.fromstring(payload)
            break
        except (OSError, ET.ParseError) as error:
            last_error = error
            retryable = not isinstance(error, HTTPError) or error.code in {429, 502, 503, 504}
            if attempt < max_retries and retryable:
                retry_after = error.headers.get("Retry-After") if isinstance(error, HTTPError) else None
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 0.5 * (2**attempt)
                time.sleep(min(delay, 4.0))
                continue
            cached = _read_cache(cache_path)
            if cached is not None:
                return cached
            fallback = _search_openalex(query, max_results, timeout_seconds)
            if fallback:
                _write_cache(cache_path, fallback)
                return fallback
            raise ArxivSearchError(f"arXiv search unavailable: {error}") from error

    if root is None:
        cached = _read_cache(cache_path)
        if cached is not None:
            return cached
        raise ArxivSearchError(f"arXiv search unavailable: {last_error}")

    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    papers: list[ParsedPaper] = []
    for entry in root.findall("atom:entry", namespace):
        raw_id = (entry.findtext("atom:id", default="", namespaces=namespace)).rstrip("/").split("/")[-1]
        paper_id = f"arxiv-{raw_id.replace('.', '-') }"
        title = " ".join(entry.findtext("atom:title", default=raw_id, namespaces=namespace).split())
        abstract = " ".join(entry.findtext("atom:summary", default="", namespaces=namespace).split())
        published = entry.findtext("atom:published", default="", namespaces=namespace)
        try:
            year = datetime.fromisoformat(published.replace("Z", "+00:00")).year
        except ValueError:
            year = None
        authors = [
            " ".join(author.findtext("atom:name", default="", namespaces=namespace).split())
            for author in entry.findall("atom:author", namespace)
        ]
        paper = Paper(
            id=paper_id,
            title=title,
            authors=[author for author in authors if author],
            year=year,
            abstract=abstract,
            source_path=f"https://arxiv.org/abs/{raw_id}",
        )
        papers.append(build_document(paper, [("Abstract", abstract, None)]))
    if papers:
        _write_cache(cache_path, papers)
    return papers


def _cache_path(cache_dir: Path, query: str, max_results: int) -> Path:
    key = hashlib.sha256(f"{query}\0{max_results}".encode()).hexdigest()
    return cache_dir / f"{key}.json"


def _read_cache(cache_path: Path | None) -> list[ParsedPaper] | None:
    if cache_path is None or not cache_path.exists():
        return None
    try:
        values = json.loads(cache_path.read_text(encoding="utf-8"))
        return [ParsedPaper.model_validate(value) for value in values]
    except (OSError, ValueError, TypeError):
        return None


def _write_cache(cache_path: Path | None, papers: list[ParsedPaper]) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps([paper.model_dump(mode="json") for paper in papers], ensure_ascii=False),
        encoding="utf-8",
    )


def _search_openalex(query: str, max_results: int, timeout_seconds: float) -> list[ParsedPaper]:
    """Use OpenAlex only as a metadata mirror when the official arXiv API is rate limited."""
    parameters = urllib.parse.urlencode(
        {
            "search": query,
            "filter": "locations.source.issn:2331-8422",
            "per-page": max_results,
        }
    )
    request = urllib.request.Request(
        f"https://api.openalex.org/works?{parameters}",
        headers={"User-Agent": "PaperScout/0.1 (research literature search)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            results = json.loads(response.read()).get("results", [])
    except (OSError, ValueError, TypeError):
        return []

    papers = []
    for result in results:
        locations = result.get("locations") or []
        arxiv_url = next(
            (
                location.get("landing_page_url")
                for location in locations
                if "arxiv.org/abs/" in (location.get("landing_page_url") or "")
            ),
            None,
        )
        if not arxiv_url:
            continue
        raw_id = arxiv_url.rstrip("/").split("/")[-1]
        inverted = result.get("abstract_inverted_index") or {}
        words = sorted(
            ((position, word) for word, positions in inverted.items() for position in positions),
            key=lambda pair: pair[0],
        )
        abstract = " ".join(word for _, word in words)
        paper = Paper(
            id=f"arxiv-{raw_id.replace('.', '-')}",
            title=result.get("title") or raw_id,
            authors=[
                item.get("author", {}).get("display_name")
                for item in result.get("authorships", [])
                if item.get("author", {}).get("display_name")
            ],
            year=result.get("publication_year"),
            abstract=abstract,
            source_path=f"https://arxiv.org/abs/{raw_id}",
        )
        papers.append(build_document(paper, [("Abstract", abstract, None)]))
    return papers
