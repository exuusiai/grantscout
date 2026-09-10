from paperscout.models.schemas import Paper, PaperSection
from paperscout.retrieval.store import CorpusStore


def get_paper_metadata(store: CorpusStore, paper_id: str) -> Paper:
    paper = store.get_paper(paper_id)
    if paper is None:
        raise KeyError(f"Unknown paper: {paper_id}")
    return paper


def get_paper_sections(store: CorpusStore, paper_id: str) -> list[PaperSection]:
    if store.get_paper(paper_id) is None:
        raise KeyError(f"Unknown paper: {paper_id}")
    return store.get_sections(paper_id)
