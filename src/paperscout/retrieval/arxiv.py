from __future__ import annotations

import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

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
) -> list[ParsedPaper]:
    query = build_arxiv_query(question)
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
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read()
        root = ET.fromstring(payload)
    except (OSError, ET.ParseError) as error:
        raise ArxivSearchError(f"arXiv search unavailable: {error}") from error

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
    return papers
