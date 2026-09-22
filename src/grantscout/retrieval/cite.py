"""Citation formatting for arXiv papers (deterministic, no LLM)."""

import re

from grantscout.models.schemas import Paper


def _bibtex_key(paper: Paper) -> str:
    author = paper.authors[0] if paper.authors else "unknown"
    surname = re.sub(r"[^A-Za-z\u3400-\u9fff]", "", author.split()[-1]).lower() or "unknown"
    first_word = re.match(r"[A-Za-z\u3400-\u9fff]+", paper.title or "untitled")
    return f"{surname}{paper.year or ''}{(first_word.group(0).lower() if first_word else 'untitled')}"


def to_bibtex(paper: Paper) -> str:
    authors = " and ".join(paper.authors) or "Unknown"
    fields = [
        f"title={{{paper.title}}}",
        f"author={{{authors}}}",
        f"year={{{paper.year or ''}}}",
    ]
    if paper.source_path:
        fields.append(f"url={{{paper.source_path}}}")
    return f"@misc{{{_bibtex_key(paper)},\n{',\n'.join(fields)}\n}}"


def to_gbt7714(paper: Paper) -> str:
    authors = paper.authors
    if len(authors) > 3:
        shown = ", ".join(authors[:3]) + ", 等"
    elif authors:
        shown = ", ".join(authors)
    else:
        shown = "佚名" if not paper.authors else ", ".join(authors)
    return f"{shown}. {paper.title}[EB/OL]. {paper.year or ''}."
