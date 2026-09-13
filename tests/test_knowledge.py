import time

from paperscout.knowledge import KnowledgeService
from paperscout.retrieval.store import CorpusStore


def test_project_upload_is_asynchronous_and_hot_indexed(tmp_path) -> None:
    service = KnowledgeService(tmp_path, workers=1)
    project = service.create_project("My Research")
    task = service.enqueue(project["id"], "notes.md", b"# Guide\n\nRetrieval batching reduces latency.")

    assert task["status"] in {"queued", "parsing", "indexing", "ready"}
    for _ in range(100):
        task = service.document(task["id"])
        if task["status"] in {"ready", "failed"}:
            break
        time.sleep(0.01)

    assert task["status"] == "ready"
    with CorpusStore(service.corpus_path(project["id"])) as store:
        assert store.paper_count() == 1
        assert store.search("batching")
    assert len(service.export(project["id"])["papers"]) == 1


def test_projects_have_isolated_corpora(tmp_path) -> None:
    service = KnowledgeService(tmp_path)
    first = service.create_project("First")
    second = service.create_project("Second")

    assert service.corpus_path(first["id"]) != service.corpus_path(second["id"])
