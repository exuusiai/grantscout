import re


def split_paragraphs(text: str, max_chars: int = 1600) -> list[str]:
    """Split text into stable, bounded evidence-sized chunks."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    chunks: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
            continue
        start = 0
        while start < len(paragraph):
            end = min(start + max_chars, len(paragraph))
            if end < len(paragraph):
                boundary = paragraph.rfind(" ", start, end)
                if boundary > start + max_chars // 2:
                    end = boundary
            chunks.append(paragraph[start:end].strip())
            start = end
    return chunks
