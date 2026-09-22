"""PII scrubbing for private-domain documents.

Two layers:

1. Rule engine (always available): Chinese phone numbers, national ID cards,
   emails, plus caller-supplied sensitive words.
2. Optional Presidio NER (``presidio-analyzer`` + a spaCy ``zh`` model):
   replaces PERSON / LOCATION / ORGANIZATION mentions. If either dependency
   is missing the layer degrades silently to rules-only — scrubbing never
   blocks ingestion.

The original file stays untouched on disk; scrubbing applies to the parsed
text that enters the search index, matching the design rule "原文保留私有区,
索引存脱敏文本".
"""

import logging
import re
from functools import lru_cache

from grantscout.models.schemas import ParsedPaper

logger = logging.getLogger(__name__)

_ID_CARD_PATTERN = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
_PHONE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

_PRESIDIO_ENTITIES = ["PERSON", "LOCATION", "ORGANIZATION"]


def scrub_rules(text: str, extra_words: list[str] | tuple[str, ...] = ()) -> str:
    if not text:
        return text
    scrubbed = _ID_CARD_PATTERN.sub("<ID_CARD>", text)
    scrubbed = _PHONE_PATTERN.sub("<PHONE>", scrubbed)
    scrubbed = _EMAIL_PATTERN.sub("<EMAIL>", scrubbed)
    for word in extra_words:
        if word:
            scrubbed = scrubbed.replace(word, "<CUSTOM>")
    return scrubbed


@lru_cache(maxsize=1)
def _presidio_analyzer():
    """Build a zh-language Presidio analyzer once; None when unavailable."""
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "zh", "model_name": "zh_core_web_sm"}],
            }
        )
        return AnalyzerEngine(
            nlp_engine=provider.create_engine(), supported_languages=["zh"]
        )
    except Exception as error:  # noqa: BLE001 - optional dependency chain
        logger.info("Presidio NER unavailable; PII falls back to rules only: %s", error)
        return None


def scrub_text(text: str, extra_words: list[str] | tuple[str, ...] = (), use_presidio: bool = True) -> str:
    if not text:
        return text
    scrubbed = scrub_rules(text, extra_words)
    if not use_presidio:
        return scrubbed
    analyzer = _presidio_analyzer()
    if analyzer is None:
        return scrubbed
    findings = analyzer.analyze(text=scrubbed, language="zh", entities=_PRESIDIO_ENTITIES)
    for finding in sorted(findings, key=lambda item: item.start, reverse=True):
        scrubbed = scrubbed[: finding.start] + f"<{finding.entity_type}>" + scrubbed[finding.end :]
    return scrubbed


def scrub_parsed_paper(
    parsed: ParsedPaper, extra_words: list[str] | tuple[str, ...] = (), use_presidio: bool = True
) -> ParsedPaper:
    paper = parsed.paper.model_copy(
        update={"abstract": scrub_text(parsed.paper.abstract, extra_words, use_presidio)}
    )
    sections = [
        section.model_copy(update={"text": scrub_text(section.text, extra_words, use_presidio)})
        for section in parsed.sections
    ]
    evidence_items = [
        item.model_copy(update={"text": scrub_text(item.text, extra_words, use_presidio)})
        for item in parsed.evidence_items
    ]
    return parsed.model_copy(update={"paper": paper, "sections": sections, "evidence_items": evidence_items})
