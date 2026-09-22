"""Persistence helpers for proposal documents stored under the runs directory."""

import json
from datetime import datetime, timezone
from pathlib import Path

from grantscout.proposal.schemas import ProposalDocument


class ProposalStore:
    """Reads and writes ``{id}.proposal.json`` artifacts managed by the pipeline."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, proposal_id: str) -> Path:
        return self.root / f"{proposal_id}.proposal.json"

    def save(self, document: ProposalDocument) -> ProposalDocument:
        document.updated_at = datetime.now(timezone.utc)
        self._path(document.id).write_text(
            json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return document

    def get(self, proposal_id: str) -> ProposalDocument:
        path = self._path(proposal_id)
        if not path.exists():
            raise KeyError(proposal_id)
        return ProposalDocument.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def list(self) -> list[dict]:
        items = []
        for path in sorted(self.root.glob("*.proposal.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                document = ProposalDocument.model_validate(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
            items.append(
                {
                    "id": document.id,
                    "title": document.title,
                    "template_id": document.template_id,
                    "status": document.status,
                    "outline_approved": document.outline.approved if document.outline else False,
                    "created_at": document.created_at.isoformat(),
                    "updated_at": document.updated_at.isoformat(),
                }
            )
        return items
