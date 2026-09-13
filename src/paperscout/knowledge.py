from __future__ import annotations

import hashlib
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Callable
from uuid import uuid4

from paperscout.models.schemas import ParsedPaper
from paperscout.retrieval.parser import parse_document
from paperscout.retrieval.store import CorpusStore

Parser = Callable[[Path, str], ParsedPaper]
PARSERS: dict[str, Parser] = {}


def register_parser(suffix: str, parser: Parser) -> None:
    PARSERS[suffix.lower().lstrip(".")] = parser


class KnowledgeService:
    def __init__(self, root: Path, workers: int = 2) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = root / "knowledge.sqlite"
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ingest")
        self.lock = Lock()
        with self._db() as db:
            db.executescript("""
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS projects(id TEXT PRIMARY KEY,name TEXT NOT NULL,created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY,project_id TEXT NOT NULL,name TEXT NOT NULL,status TEXT NOT NULL,progress INTEGER NOT NULL,error TEXT,path TEXT,sha256 TEXT,created_at TEXT NOT NULL,FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE);
            """)

    def _db(self):
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        return db

    def create_project(self, name: str) -> dict:
        project_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower() or uuid4().hex[:12]
        with self._db() as db:
            db.execute("INSERT OR IGNORE INTO projects VALUES(?,?,?)", (project_id, name, _now()))
        return {"id": project_id, "name": name}

    def projects(self) -> list[dict]:
        with self._db() as db:
            return [dict(row) for row in db.execute("SELECT * FROM projects ORDER BY created_at DESC")]

    def enqueue(self, project_id: str, name: str, payload: bytes) -> dict:
        self._require(project_id)
        suffix = Path(name).suffix.lower()
        if suffix not in {".md", ".markdown", ".pdf", ".pptx", ".txt"} and suffix[1:] not in PARSERS:
            raise ValueError(f"Unsupported format: {suffix}")
        document_id = uuid4().hex
        directory = self.root / "projects" / project_id / "uploads"
        directory.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", Path(name).name)
        path = directory / f"{document_id}-{safe_name}"
        path.write_bytes(payload)
        with self._db() as db:
            db.execute("INSERT INTO documents VALUES(?,?,?,?,?,?,?,?,?)", (document_id, project_id, name, "queued", 0, None, str(path), hashlib.sha256(payload).hexdigest(), _now()))
        self.pool.submit(self._ingest, document_id, project_id, path)
        return self.document(document_id)

    def _ingest(self, document_id: str, project_id: str, path: Path) -> None:
        self._update(document_id, "parsing", 25)
        try:
            parser = PARSERS.get(path.suffix.lower().lstrip("."))
            parsed = parser(path, document_id) if parser else parse_document(path, paper_id=document_id)
            self._update(document_id, "indexing", 70)
            with self.lock, CorpusStore(self.corpus_path(project_id)) as store:
                store.upsert(parsed)
            self._update(document_id, "ready", 100)
        except Exception as error:
            self._update(document_id, "failed", 100, str(error)[:1000])

    def _update(self, document_id: str, status: str, progress: int, error: str | None = None) -> None:
        with self._db() as db:
            db.execute("UPDATE documents SET status=?,progress=?,error=? WHERE id=?", (status, progress, error, document_id))

    def document(self, document_id: str) -> dict:
        with self._db() as db:
            row = db.execute("SELECT * FROM documents WHERE id=?", (document_id,)).fetchone()
        if not row:
            raise KeyError(document_id)
        return dict(row)

    def documents(self, project_id: str) -> list[dict]:
        self._require(project_id)
        with self._db() as db:
            return [dict(row) for row in db.execute("SELECT * FROM documents WHERE project_id=? ORDER BY created_at DESC", (project_id,))]

    def corpus_path(self, project_id: str) -> Path:
        return self.root / "projects" / project_id / "corpus.sqlite"

    def export(self, project_id: str) -> dict:
        self._require(project_id)
        path = self.corpus_path(project_id)
        if not path.exists():
            return {"project_id": project_id, "papers": []}
        with CorpusStore(path) as store:
            return {"project_id": project_id, "papers": [paper.model_dump(mode="json") for paper in store.list_papers()]}

    def _require(self, project_id: str) -> None:
        with self._db() as db:
            if not db.execute("SELECT 1 FROM projects WHERE id=?", (project_id,)).fetchone():
                raise KeyError(project_id)


def _now() -> str:
    return datetime.now(UTC).isoformat()
