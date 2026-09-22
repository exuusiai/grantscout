"""Sentence-level index for verbatim (Type C) completion.

Splits stored evidence into sentences with char offsets and indexes them in
an FTS5 ``trigram`` table so a user's typed fragment can find the exact
library sentence that continues it — zero LLM, millisecond latency.
Falls back to ``LIKE`` scanning when the trigram tokenizer is unavailable.
"""

import logging
import re
from dataclasses import dataclass

from grantscout.models.schemas import EvidenceItem
from grantscout.retrieval.store import CorpusStore

logger = logging.getLogger(__name__)

_SENTENCE_PATTERN = re.compile(r"[^。！？!?]+[。！？!?.]?|[^。！？!?]+$")
_MIN_SENTENCE_LEN = 8
_MIN_FRAGMENT_LEN = 3


@dataclass(frozen=True)
class SentenceHit:
    sentence: str
    remainder: str
    evidence: EvidenceItem
    char_start: int


class SentenceIndex:
    """Trigram-backed sentence continuation lookup over one corpus."""

    def __init__(self, store: CorpusStore) -> None:
        self.store = store
        self._ensured = False
        self._trigram = True

    def ensure(self) -> None:
        """Create tables and populate them once per corpus (idempotent)."""
        if self._ensured:
            return
        connection = self.store.connection
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence_sentences (
                    sentence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    evidence_id TEXT NOT NULL,
                    paper_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS sentence_evidence_idx ON evidence_sentences(evidence_id)"
            )
            try:
                connection.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS sentence_fts USING fts5(text, tokenize='trigram')"
                )
                self._trigram = True
            except Exception as error:  # noqa: BLE001 - trigram is version dependent
                logger.warning("Trigram tokenizer unavailable; verbatim uses LIKE: %s", error)
                self._trigram = False
            count = int(connection.execute("SELECT COUNT(*) FROM evidence_sentences").fetchone()[0])
            if count == 0:
                rows = connection.execute("SELECT id, paper_id, text FROM evidence").fetchall()
                for evidence_id, paper_id, text in rows:
                    for sentence, start, end in split_sentences(text):
                        connection.execute(
                            "INSERT INTO evidence_sentences(evidence_id, paper_id, text, char_start, char_end)"
                            " VALUES (?, ?, ?, ?, ?)",
                            (evidence_id, paper_id, sentence, start, end),
                        )
                        if self._trigram:
                            connection.execute(
                                "INSERT INTO sentence_fts VALUES (?)", (sentence,)
                            )
                built = int(connection.execute("SELECT COUNT(*) FROM evidence_sentences").fetchone()[0])
                logger.info("Sentence index built: %d sentences", built)
        self._ensured = True

    def continue_fragment(
        self, prefix: str, top_k: int = 3, paper_ids: set[str] | None = None
    ) -> list[SentenceHit]:
        """Find library sentences that verbatim-continue the typed fragment.

        Tries progressively shorter trailing fragments so user lead-ins
        ("根据调研,EMANE是由…") still find the library sentence.
        """
        self.ensure()
        tail = prefix.rstrip()
        fragments: list[str] = []
        for length in (24, 16, 12, 8, 6, 4):
            fragment = tail[-length:].lstrip("。!?;;,,").strip()
            if len(fragment) < _MIN_FRAGMENT_LEN or fragment in fragments:
                continue
            fragments.append(fragment)
        for fragment in fragments:
            hits = self._query_fragment(fragment, top_k, paper_ids)
            if hits:
                return hits
        return []

    def _query_fragment(
        self, fragment: str, top_k: int, paper_ids: set[str] | None
    ) -> list[SentenceHit]:
        connection = self.store.connection
        candidates: list[tuple[str, str]] = []  # (evidence_id, sentence)
        if self._trigram:
            try:
                sql = (
                    "SELECT evidence_sentences.evidence_id, evidence_sentences.text "
                    "FROM sentence_fts JOIN evidence_sentences "
                    "ON sentence_fts.rowid = evidence_sentences.sentence_id "
                    "WHERE sentence_fts MATCH ? LIMIT ?"
                )
                rows = connection.execute(sql, (f'"{_escape_quotes(fragment)}"', top_k * 8)).fetchall()
                candidates = [(row[0], row[1]) for row in rows]
            except Exception as error:  # noqa: BLE001 - malformed queries fall back
                logger.warning("Trigram continuation query failed: %s", error)
        if not candidates and not self._trigram:
            sql = "SELECT evidence_id, text FROM evidence_sentences WHERE text LIKE ? LIMIT ?"
            rows = connection.execute(sql, (f"%{fragment}%", top_k * 8)).fetchall()
            candidates = [(row[0], row[1]) for row in rows]
        hits: list[SentenceHit] = []
        seen: set[str] = set()
        for evidence_id, sentence in candidates:
            index = sentence.find(fragment)
            if index < 0:
                continue
            remainder = sentence[index + len(fragment):].strip()
            if len(remainder) < 4 or sentence in seen:
                continue
            evidence = self.store.get_evidence(evidence_id)
            if evidence is None:
                continue
            if paper_ids is not None and evidence.paper_id not in paper_ids:
                continue
            seen.add(sentence)
            char_start = evidence.start_char + index
            hits.append(
                SentenceHit(
                    sentence=sentence,
                    remainder=remainder,
                    evidence=evidence,
                    char_start=char_start,
                )
            )
            if len(hits) >= top_k:
                break
        return hits


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Split into sentences, returning (sentence, char_start, char_end)."""
    results: list[tuple[str, int, int]] = []
    for match in _SENTENCE_PATTERN.finditer(text):
        raw = match.group()
        stripped = raw.strip()
        if len(stripped) < _MIN_SENTENCE_LEN:
            continue
        offset = raw.find(stripped)
        results.append((stripped, match.start() + offset, match.start() + offset + len(stripped)))
    return results


def _trailing_fragment(prefix: str) -> str:
    """Take the trailing few characters of the prefix as the lookup fragment."""
    tail = prefix.rstrip()[-24:]
    tail = tail.lstrip("。！？!?;；,， ")
    return tail.strip()


def _escape_quotes(value: str) -> str:
    return value.replace('"', '""')
