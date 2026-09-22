import json
import re
import sqlite3
from pathlib import Path

from grantscout.models.schemas import EvidenceItem, Paper, PaperSection, ParsedPaper, SearchResult


def _tokens(text: str) -> list[str]:
    # Keep Latin technical terms separate when users type them directly next to Chinese text.
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9+._-]*|[0-9]+|[\u3400-\u9fff]+", text)
    return [token.lower() for token in tokens if len(token) > 1]


class CorpusStore:
    """SQLite-backed corpus with transparent lexical retrieval for V1."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA busy_timeout = 5000")
        try:
            self.connection.execute("PRAGMA journal_mode = WAL")
        except sqlite3.OperationalError:
            pass
        self.fts_available = False
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "CorpusStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                authors_json TEXT NOT NULL,
                year INTEGER,
                abstract TEXT NOT NULL,
                source_path TEXT
            );
            CREATE TABLE IF NOT EXISTS sections (
                id TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                section_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                page_start INTEGER,
                page_end INTEGER
            );
            CREATE TABLE IF NOT EXISTS evidence (
                id TEXT PRIMARY KEY,
                paper_id TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
                section_id TEXT NOT NULL REFERENCES sections(id) ON DELETE CASCADE,
                text TEXT NOT NULL,
                page INTEGER,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS evidence_paper_idx ON evidence(paper_id);
            """
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS _meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        needs_fts_sync = (
            self.connection.execute("SELECT value FROM _meta WHERE key='fts_sync'").fetchone()
            is None
        )
        try:
            self.connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(
                    evidence_id UNINDEXED,
                    paper_id UNINDEXED,
                    title,
                    text,
                    tokenize='unicode61'
                )
                """
            )
            if needs_fts_sync:
                # Verify/rebuild only once per corpus; upsert keeps FTS in sync afterwards.
                evidence_count = int(
                    self.connection.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
                )
                fts_count = int(
                    self.connection.execute("SELECT COUNT(*) FROM evidence_fts").fetchone()[0]
                )
                if evidence_count != fts_count:
                    self.connection.execute("DELETE FROM evidence_fts")
                    self.connection.execute(
                        """
                        INSERT INTO evidence_fts(evidence_id, paper_id, title, text)
                        SELECT e.id, e.paper_id, p.title, e.text
                        FROM evidence e JOIN papers p ON p.id = e.paper_id
                        """
                    )
                self.connection.execute(
                    "INSERT OR REPLACE INTO _meta(key,value) VALUES('fts_sync','v1')"
                )
            self.fts_available = True
        except sqlite3.OperationalError:
            self.fts_available = False
        self._ensure_metadata_columns()
        self.connection.commit()

    def _ensure_metadata_columns(self) -> None:
        """Add paper metadata columns to corpora created before Phase 2."""
        columns = {row["name"] for row in self.connection.execute("PRAGMA table_info(papers)")}
        additions = {
            "scope": "TEXT DEFAULT 'public'",
            "venue": "TEXT",
            "language": "TEXT",
            "tags": "TEXT",
            "cited_by_count": "INTEGER",
        }
        for name, declaration in additions.items():
            if name not in columns:
                self.connection.execute(f"ALTER TABLE papers ADD COLUMN {name} {declaration}")

    def set_paper_meta(
        self,
        paper_id: str,
        *,
        scope: str | None = None,
        venue: str | None = None,
        language: str | None = None,
        tags: list[str] | None = None,
        cited_by_count: int | None = None,
    ) -> None:
        if self.get_paper(paper_id) is None:
            raise ValueError(f"Unknown paper: {paper_id}")
        assignments, parameters = [], []
        for column, value in (
            ("scope", scope),
            ("venue", venue),
            ("language", language),
            ("tags", json.dumps(tags, ensure_ascii=False) if tags is not None else None),
            ("cited_by_count", cited_by_count),
        ):
            if value is not None:
                assignments.append(f"{column} = ?")
                parameters.append(value)
        if not assignments:
            return
        parameters.append(paper_id)
        with self.connection:
            self.connection.execute(
                f"UPDATE papers SET {', '.join(assignments)} WHERE id = ?", parameters
            )

    def get_paper_meta(self, paper_id: str) -> dict:
        row = self.connection.execute(
            "SELECT scope, venue, language, tags, cited_by_count FROM papers WHERE id = ?",
            (paper_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown paper: {paper_id}")
        return {
            "paper_id": paper_id,
            "scope": row["scope"] or "public",
            "venue": row["venue"],
            "language": row["language"],
            "tags": json.loads(row["tags"]) if row["tags"] else [],
            "cited_by_count": row["cited_by_count"],
        }

    def list_paper_meta(self) -> dict[str, dict]:
        rows = self.connection.execute(
            "SELECT id, scope, venue, language, tags, cited_by_count FROM papers"
        ).fetchall()
        return {
            row["id"]: {
                "paper_id": row["id"],
                "scope": row["scope"] or "public",
                "venue": row["venue"],
                "language": row["language"],
                "tags": json.loads(row["tags"]) if row["tags"] else [],
                "cited_by_count": row["cited_by_count"],
            }
            for row in rows
        }

    def upsert(self, document: ParsedPaper) -> None:
        paper = document.paper
        with self.connection:
            if self.fts_available:
                self.connection.execute("DELETE FROM evidence_fts WHERE paper_id = ?", (paper.id,))
            self.connection.execute("DELETE FROM papers WHERE id = ?", (paper.id,))
            self.connection.execute(
                """
                INSERT INTO papers (id, title, authors_json, year, abstract, source_path)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    paper.id,
                    paper.title,
                    json.dumps(paper.authors, ensure_ascii=False),
                    paper.year,
                    paper.abstract,
                    paper.source_path,
                ),
            )
            self.connection.executemany(
                "INSERT INTO sections VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        section.id,
                        section.paper_id,
                        section.title,
                        section.section_index,
                        section.text,
                        section.page_start,
                        section.page_end,
                    )
                    for section in document.sections
                ],
            )
            if self.fts_available:
                self.connection.executemany(
                    "INSERT INTO evidence_fts(evidence_id, paper_id, title, text) VALUES (?, ?, ?, ?)",
                    [
                        (item.id, item.paper_id, paper.title, item.text)
                        for item in document.evidence_items
                    ],
                )
            self.connection.executemany(
                "INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        item.id,
                        item.paper_id,
                        item.section_id,
                        item.text,
                        item.page,
                        item.start_char,
                        item.end_char,
                    )
                    for item in document.evidence_items
                ],
            )

    def delete(self, paper_id: str) -> None:
        with self.connection:
            if self.fts_available:
                self.connection.execute("DELETE FROM evidence_fts WHERE paper_id = ?", (paper_id,))
            self.connection.execute("DELETE FROM papers WHERE id = ?", (paper_id,))

    def paper_count(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS count FROM papers").fetchone()
        return int(row["count"])

    def list_papers(self) -> list[Paper]:
        rows = self.connection.execute("SELECT * FROM papers ORDER BY id").fetchall()
        return [self._paper_from_row(row) for row in rows]

    def get_paper(self, paper_id: str) -> Paper | None:
        row = self.connection.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        return self._paper_from_row(row) if row else None

    def get_sections(self, paper_id: str) -> list[PaperSection]:
        rows = self.connection.execute(
            "SELECT * FROM sections WHERE paper_id = ? ORDER BY section_index", (paper_id,)
        ).fetchall()
        return [
            PaperSection(
                id=row["id"],
                paper_id=row["paper_id"],
                title=row["title"],
                section_index=row["section_index"],
                text=row["text"],
                page_start=row["page_start"],
                page_end=row["page_end"],
            )
            for row in rows
        ]

    def get_evidence(self, evidence_id: str) -> EvidenceItem | None:
        row = self.connection.execute("SELECT * FROM evidence WHERE id = ?", (evidence_id,)).fetchone()
        return self._evidence_from_row(row) if row else None

    def list_evidence(self) -> list[tuple[Paper, EvidenceItem]]:
        rows = self.connection.execute(
            """
            SELECT e.*, p.id AS parent_paper_id, p.title, p.authors_json, p.year, p.abstract, p.source_path
            FROM evidence e JOIN papers p ON p.id = e.paper_id
            ORDER BY e.paper_id, e.id
            """
        ).fetchall()
        return [(self._paper_from_row(row), self._evidence_from_row(row)) for row in rows]

    def search(
        self, query: str, top_k: int = 10, paper_ids: set[str] | None = None
    ) -> list[SearchResult]:
        query_terms = set(_tokens(query))
        if not query_terms:
            return []
        if self.fts_available:
            return self._search_fts(query_terms, top_k, paper_ids)
        return self._search_scan(query, query_terms, top_k, paper_ids)

    def _search_fts(
        self, query_terms: set[str], top_k: int, paper_ids: set[str] | None = None
    ) -> list[SearchResult]:
        match_query = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in query_terms)
        paper_filter = ""
        parameters: list[object] = [match_query]
        if paper_ids is not None:
            paper_filter = f" AND e.paper_id IN ({', '.join('?' for _ in paper_ids)})"
            parameters.extend(sorted(paper_ids))
        parameters.append(top_k)
        rows = self.connection.execute(
            f"""
            SELECT e.*, p.id AS parent_paper_id, p.title, p.authors_json, p.year,
                   p.abstract, p.source_path, bm25(evidence_fts, 0.0, 0.0, 2.0, 1.0) AS rank
            FROM evidence_fts
            JOIN evidence e ON e.id = evidence_fts.evidence_id
            JOIN papers p ON p.id = e.paper_id
            WHERE evidence_fts MATCH ?{paper_filter}
            ORDER BY rank
            LIMIT ?
            """,
            parameters,
        ).fetchall()
        results: list[SearchResult] = []
        for row in rows:
            haystack = f"{row['title']} {row['text']}".lower()
            matched_terms = sorted(term for term in query_terms if term in haystack)
            rank = abs(float(row["rank"]))
            results.append(
                SearchResult(
                    evidence=self._evidence_from_row(row),
                    paper=self._paper_from_row(row),
                    score=round(rank / (1.0 + rank), 6),
                    matched_terms=matched_terms,
                )
            )
        return results

    def _search_scan(
        self, query: str, query_terms: set[str], top_k: int, paper_ids: set[str] | None = None
    ) -> list[SearchResult]:
        rows = self.connection.execute(
            """
            SELECT e.*, p.id AS parent_paper_id, p.title, p.authors_json, p.year, p.abstract, p.source_path
            FROM evidence e JOIN papers p ON p.id = e.paper_id
            """
        ).fetchall()
        results: list[SearchResult] = []
        for row in rows:
            if paper_ids is not None and row["paper_id"] not in paper_ids:
                continue
            haystack = f"{row['title']} {row['text']}".lower()
            matched_terms = sorted(term for term in query_terms if term in haystack)
            if not matched_terms:
                continue
            score = sum(haystack.count(term) for term in matched_terms) / max(len(_tokens(row["text"])), 1)
            score += 0.5 * len(matched_terms) / len(query_terms)
            if query.lower() in haystack:
                score += 1.0
            results.append(
                SearchResult(
                    evidence=self._evidence_from_row(row),
                    paper=self._paper_from_row(row),
                    score=round(score, 6),
                    matched_terms=matched_terms,
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    @staticmethod
    def _paper_from_row(row: sqlite3.Row) -> Paper:
        return Paper(
            id=row["parent_paper_id"] if "parent_paper_id" in row.keys() else row["id"],
            title=row["title"],
            authors=json.loads(row["authors_json"]),
            year=row["year"],
            abstract=row["abstract"],
            source_path=row["source_path"],
        )

    @staticmethod
    def _evidence_from_row(row: sqlite3.Row) -> EvidenceItem:
        return EvidenceItem(
            id=row["id"],
            paper_id=row["paper_id"],
            section_id=row["section_id"],
            text=row["text"],
            page=row["page"],
            start_char=row["start_char"],
            end_char=row["end_char"],
        )
