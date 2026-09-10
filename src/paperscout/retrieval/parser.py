import re
from pathlib import Path

from paperscout.models.schemas import EvidenceItem, Paper, PaperSection, ParsedPaper
from paperscout.retrieval.chunker import split_paragraphs


def _safe_id(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")
    return normalized or "paper"


def _section_blocks(text: str) -> list[tuple[str, str]]:
    lines = text.replace("\r\n", "\n").split("\n")
    numbered_heading = re.compile(r"^\d+(?:\.\d+)*[.)]?\s+\S")
    known_headings = {
        "abstract",
        "introduction",
        "background",
        "method",
        "methods",
        "methodology",
        "dataset",
        "datasets",
        "experiments",
        "evaluation",
        "results",
        "discussion",
        "conclusion",
        "conclusions",
        "limitations",
        "references",
    }

    def is_heading(line: str) -> bool:
        if line.startswith("#"):
            return True
        if numbered_heading.match(line):
            return True
        if line.lower() in known_headings:
            return True
        return len(line) > 3 and line.isupper() and len(line) <= 100

    blocks: list[tuple[str, list[str]]] = [("Document", [])]
    for line in lines:
        stripped = line.strip()
        if stripped and is_heading(stripped.lstrip("#").strip()) and len(stripped.split()) <= 14:
            blocks.append((stripped, []))
        else:
            blocks[-1][1].append(line)
    return [(title, "\n".join(content).strip()) for title, content in blocks if "\n".join(content).strip()]


def build_document(
    paper: Paper,
    section_inputs: list[tuple[str, str, int | None]],
) -> ParsedPaper:
    sections: list[PaperSection] = []
    evidence_items: list[EvidenceItem] = []
    for section_index, (title, text, page) in enumerate(section_inputs):
        section_id = f"{paper.id}:section:{section_index:04d}"
        section = PaperSection(
            id=section_id,
            paper_id=paper.id,
            title=title,
            section_index=section_index,
            text=text,
            page_start=page,
            page_end=page,
        )
        sections.append(section)
        cursor = 0
        for evidence_index, chunk in enumerate(split_paragraphs(text)):
            start_char = text.find(chunk, cursor)
            if start_char < 0:
                start_char = cursor
            end_char = start_char + len(chunk)
            evidence_items.append(
                EvidenceItem(
                    id=f"{section_id}:evidence:{evidence_index:04d}",
                    paper_id=paper.id,
                    section_id=section_id,
                    text=chunk,
                    page=page,
                    start_char=start_char,
                    end_char=end_char,
                )
            )
            cursor = end_char
    return ParsedPaper(paper=paper, sections=sections, evidence_items=evidence_items)


def parse_document(
    path: Path,
    paper_id: str | None = None,
    title: str | None = None,
    year: int | None = None,
) -> ParsedPaper:
    """Parse TXT, Markdown, or PDF into stable sections and evidence items."""
    if not path.is_file():
        raise FileNotFoundError(path)
    resolved_id = _safe_id(paper_id or path.stem)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _parse_pdf(path, resolved_id, title, year)
    if suffix not in {".txt", ".md", ".markdown"}:
        raise ValueError(f"Unsupported paper format: {suffix}. Use .pdf, .txt, or .md")
    text = path.read_text(encoding="utf-8")
    sections = _section_blocks(text)
    inferred_title = title or sections[0][0] if sections else title or path.stem
    paper = Paper(
        id=resolved_id,
        title=inferred_title,
        year=year,
        source_path=str(path.resolve()),
    )
    return build_document(paper, [(section_title, section_text, None) for section_title, section_text in sections])


def _parse_pdf(path: Path, paper_id: str, title: str | None, year: int | None) -> ParsedPaper:
    try:
        import fitz
    except ImportError as error:
        raise RuntimeError("PDF support requires PyMuPDF; install the project dependencies") from error

    document = fitz.open(path)
    page_inputs: list[tuple[str, str, int | None]] = []
    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text").strip()
        if text:
            page_inputs.append((f"Page {page_number}", text, page_number))
    inferred_title = title or (page_inputs[0][1].splitlines()[0][:200] if page_inputs else path.stem)
    paper = Paper(
        id=paper_id,
        title=inferred_title,
        year=year,
        source_path=str(path.resolve()),
    )
    return build_document(paper, page_inputs)
