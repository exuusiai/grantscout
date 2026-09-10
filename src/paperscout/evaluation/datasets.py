import json
from pathlib import Path
from typing import Any, Iterator

from paperscout.models.schemas import Paper, ParsedPaper
from paperscout.retrieval.parser import build_document
from paperscout.retrieval.store import CorpusStore


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                parts.append(str(item[1]))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part.strip())
    return str(value or "")


def _section_inputs(record: dict[str, Any]) -> list[tuple[str, str, int | None]]:
    """Normalize Qasper and generic full-text shapes into parser sections."""
    full_text = record.get("full_text")
    sections: list[tuple[str, str, int | None]] = []
    if isinstance(full_text, dict):
        for section_title, section_body in full_text.items():
            text = _as_text(section_body)
            if text.strip():
                sections.append((str(section_title), text, None))
    elif isinstance(full_text, list):
        for index, section in enumerate(full_text):
            if not isinstance(section, dict):
                continue
            section_title = (
                section.get("section_name")
                or section.get("section_title")
                or section.get("title")
                or f"Section {index}"
            )
            section_body = section.get("paragraphs")
            if section_body is None:
                section_body = section.get("text") or section.get("body") or ""
            if isinstance(section_body, list):
                text = "\n\n".join(
                    part for item in section_body if (part := _as_text(item)).strip()
                )
            else:
                text = _as_text(section_body)
            if text.strip():
                sections.append((str(section_title), text, None))
    return sections


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            for item in value:
                if not isinstance(item, dict):
                    raise ValueError(f"JSON record in {path} is not an object")
                yield item
            return
        if isinstance(value, dict) and any(key in value for key in ("title", "full_text", "abstract", "text")):
            yield value
            return
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(item, dict):
                    raise ValueError(f"JSON record in {path} is not an object")
                item = dict(item)
                item.setdefault("id", key)
                yield item
            return
        raise ValueError(f"Unsupported JSON dataset root in {path}")
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            yield value


iter_jsonl = iter_records


def record_to_document(record: dict[str, Any], index: int) -> ParsedPaper:
    """Convert common Qasper, SciFact, or generic records to ParsedPaper."""
    paper_id = str(record.get("id") or record.get("paper_id") or record.get("doc_id") or f"paper-{index:06d}")
    title = str(record.get("title") or paper_id)
    raw_authors = record.get("authors") or record.get("author") or []
    authors = [str(author) for author in raw_authors] if isinstance(raw_authors, list) else [str(raw_authors)]
    year = record.get("year")
    try:
        year = int(year) if year is not None else None
    except (TypeError, ValueError):
        year = None
    sections = _section_inputs(record)
    abstract = _as_text(record.get("abstract", ""))
    if abstract.strip() and not sections:
        sections.append(("Abstract", abstract, None))
    if not sections:
        text = _as_text(record.get("text") or record.get("body") or record.get("content") or "")
        if text.strip():
            sections.append(("Document", text, None))
    paper = Paper(
        id=paper_id,
        title=title,
        authors=authors,
        year=year,
        abstract=abstract,
    )
    return build_document(paper, sections)


def ingest_jsonl(path: Path, store: CorpusStore) -> dict[str, int]:
    papers = 0
    evidence_items = 0
    for index, record in enumerate(iter_records(path)):
        document = record_to_document(record, index)
        store.upsert(document)
        papers += 1
        evidence_items += len(document.evidence_items)
    return {"papers": papers, "evidence_items": evidence_items}


def evaluation_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for record in iter_records(path):
        if "query" not in record:
            raise ValueError("Evaluation records require a query field")
        records.append(record)
    return records
