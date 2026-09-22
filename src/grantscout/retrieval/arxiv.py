from __future__ import annotations

import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError

from grantscout.models.schemas import Paper, ParsedPaper
from grantscout.retrieval.parser import build_document


class ArxivSearchError(RuntimeError):
    """Raised when the official arXiv API cannot be queried or parsed."""


def build_arxiv_query(question: str) -> str:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+._-]{1,}", question)
    ignored = {"about", "article", "articles", "find", "paper", "papers", "please", "related", "search"}
    selected = [token for token in tokens if token.lower() not in ignored]
    if selected:
        return " ".join(dict.fromkeys(selected))
    glossary = {
        "世界模型": "world model",
        "多模态": "multimodal",
        "大语言模型": "large language model",
        "语言模型": "language model",
        "强化学习": "reinforcement learning",
        "扩散模型": "diffusion model",
        "生成模型": "generative model",
        "视觉语言": "vision language",
        "智能体": "AI agent",
        "检索增强生成": "retrieval augmented generation",
        "知识图谱": "knowledge graph",
        "机器翻译": "machine translation",
        "目标检测": "object detection",
        "图像生成": "image generation",
    }
    translated = []
    covered: set[int] = set()
    for chinese, english in sorted(glossary.items(), key=lambda item: len(item[0]), reverse=True):
        start = question.find(chinese)
        if start < 0:
            continue
        positions = set(range(start, start + len(chinese)))
        if positions & covered:
            continue
        translated.append(english)
        covered.update(positions)
    return " ".join(dict.fromkeys(translated)) or question.strip()


def search_arxiv(
    question: str,
    max_results: int = 20,
    timeout_seconds: float = 12.0,
    api_url: str = "https://export.arxiv.org/api/query",
    cache_dir: Path | None = None,
    max_retries: int = 2,
    ranking: str = "relevance",
    search_query: str | None = None,
    prefer_mirror: bool = True,
) -> list[ParsedPaper]:
    query = search_query or build_arxiv_query(question)
    cache_path = _cache_path(cache_dir, query, max_results, ranking) if cache_dir else None
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached
    # AutoDL routes to export.arxiv.org are frequently slow. OpenAlex indexes
    # arXiv metadata and lets us return only records with canonical arXiv URLs;
    # the official Atom API remains the fallback when the mirror has no result.
    if prefer_mirror:
        mirrored = _search_openalex(query, max_results, min(timeout_seconds, 5.0), ranking)
        if mirrored:
            _write_cache(cache_path, mirrored)
            return mirrored
        web_results = _search_arxiv_html(query, max_results, min(timeout_seconds, 8.0), ranking)
        if web_results:
            _write_cache(cache_path, web_results)
            return web_results
        return []
    parameters = urllib.parse.urlencode(
        {
            "search_query": (
                f'all:"{query.replace(chr(34), "")}" AND '
                "(cat:cs.AI OR cat:cs.LG OR cat:cs.CV OR cat:cs.CL OR cat:cs.RO)"
            ),
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate" if ranking == "recent" else "relevance",
            "sortOrder": "descending",
        }
    )
    last_error: Exception | None = None
    root = None
    api_urls = [api_url]
    alternate = "https://arxiv.org/api/query"
    if api_url.rstrip("/") != alternate:
        api_urls.append(alternate)
    for attempt in range(max_retries + 1):
        endpoint = api_urls[min(attempt, len(api_urls) - 1)]
        request = urllib.request.Request(
            f"{endpoint}?{parameters}",
            headers={"User-Agent": "GrantScout/0.1 (research literature search)"},
        )
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
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 0.25 * (2**attempt)
                time.sleep(min(delay, 4.0))
                continue
            cached = _read_cache(cache_path)
            if cached is not None:
                return cached
            fallback = _search_arxiv_html(query, max_results, timeout_seconds, ranking)
            if fallback:
                _write_cache(cache_path, fallback)
                return fallback
            fallback = _search_openalex(query, max_results, timeout_seconds, ranking)
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


def _search_arxiv_html(
    query: str, max_results: int, timeout_seconds: float, ranking: str = "relevance"
) -> list[ParsedPaper]:
    """Parse arXiv's public search page when the Atom API is rate limited."""
    search_parameters = {
        "query": query,
        "searchtype": "all",
        "abstracts": "show",
        "size": max(25, max_results),
    }
    search_parameters["order"] = "-announced_date_first"
    parameters = urllib.parse.urlencode(search_parameters)
    request = urllib.request.Request(
        f"https://arxiv.org/search/?{parameters}",
        headers={"User-Agent": "GrantScout/0.1 (research literature search)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            document = response.read().decode("utf-8", errors="replace")
    except OSError:
        return []

    papers = []
    for block in document.split('<li class="arxiv-result">')[1:]:
        identifier = re.search(r'href="https://arxiv\.org/abs/([^"?]+)"', block)
        title = re.search(r'<p class="title is-5 mathjax">(.*?)</p>', block, re.DOTALL)
        abstract = re.search(r'<span class="abstract-full[^>]*>(.*?)<a class=', block, re.DOTALL)
        authors = re.search(r'<p class="authors">(.*?)</p>', block, re.DOTALL)
        if not identifier or not title:
            continue
        raw_id = identifier.group(1).split("v", 1)[0]
        clean_title = _strip_html(title.group(1))
        clean_abstract = _strip_html(abstract.group(1)) if abstract else ""
        author_names = re.findall(r">([^<>]+)</a>", authors.group(1)) if authors else []
        year_match = re.match(r"(\d{2})", raw_id)
        year = 2000 + int(year_match.group(1)) if year_match else None
        paper = Paper(
            id=f"arxiv-{raw_id.replace('.', '-')}", title=clean_title,
            authors=[_strip_html(name) for name in author_names], year=year,
            abstract=clean_abstract, source_path=f"https://arxiv.org/abs/{raw_id}",
        )
        papers.append(build_document(paper, [("Abstract", clean_abstract, None)]))
        if len(papers) >= max_results:
            break
    return papers


def _strip_html(value: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", value)).split())


def _cache_path(cache_dir: Path, query: str, max_results: int, ranking: str) -> Path:
    key = hashlib.sha256(f"v3\0{query}\0{max_results}\0{ranking}".encode()).hexdigest()
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


def _search_openalex(
    query: str, max_results: int, timeout_seconds: float, ranking: str = "relevance"
) -> list[ParsedPaper]:
    """Use OpenAlex only as a metadata mirror when the official arXiv API is rate limited."""
    parameters = urllib.parse.urlencode(
        {
            "search": query,
            "filter": (
                "locations.source.issn:2331-8422,"
                "primary_topic.field.id:17"
            ),
            "per-page": max_results,
            "sort": {
                "recent": "publication_date:desc",
                "citations": "cited_by_count:desc",
                "relevance": "relevance_score:desc",
            }.get(ranking, "relevance_score:desc"),
        }
    )
    request = urllib.request.Request(
        f"https://api.openalex.org/works?{parameters}",
        headers={"User-Agent": "GrantScout/0.1 (research literature search)"},
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
