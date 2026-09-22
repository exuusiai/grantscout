from __future__ import annotations

import json
import hashlib
from collections.abc import Iterator
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from pydantic import BaseModel, Field, model_validator

from grantscout.agent.conversation import (
    ConversationMessage,
    ConversationResult,
    answer_with_project_sources,
    remember_conversation,
    understand_request,
)
from grantscout.agent.loop import GrantScoutAgent
from grantscout.completion import CompletionCache, CompletionEngine
from grantscout.completion.schemas import CompletionRequest
from grantscout.config import Settings, get_settings
from grantscout.knowledge import KnowledgeService
from grantscout.models.schemas import Paper, ResearchConstraints
from grantscout.proposal.pipeline import ProposalPipeline, ProposalPipelineError
from grantscout.proposal.renderer import render_proposal_markdown
from grantscout.proposal.schemas import ProposalRequest
from grantscout.proposal.store import ProposalStore
from grantscout.reports.renderer import render_html, render_html_fragment, render_markdown
from grantscout.retrieval.arxiv import ArxivSearchError, search_arxiv
from grantscout.retrieval.cite import to_bibtex, to_gbt7714
from grantscout.retrieval.query_interpreter import interpret_query
from grantscout.retrieval.store import CorpusStore

app = FastAPI(title="GrantScout", version="0.2.0")

if get_settings().metrics_enabled:
    # 可观测性:GRANTSCOUT_METRICS_ENABLED=true 时暴露 /metrics(Prometheus 格式)。
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app)


def _knowledge() -> KnowledgeService:
    settings = get_settings()
    if not hasattr(app.state, "knowledge"):
        app.state.knowledge = KnowledgeService(Path(settings.data_dir) / "knowledge")
    return app.state.knowledge


def _proposals() -> ProposalStore:
    settings = get_settings()
    if not hasattr(app.state, "proposals"):
        app.state.proposals = ProposalStore(Path(settings.runs_dir))
    return app.state.proposals


class ProjectRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


@app.post("/api/projects")
def create_project(request: ProjectRequest) -> dict:
    return _knowledge().create_project(request.name)


@app.get("/api/projects")
def list_projects() -> list[dict]:
    return _knowledge().projects()


@app.delete("/api/projects/{project_id}", status_code=204)
def delete_project(project_id: str) -> None:
    try:
        _knowledge().delete_project(project_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Unknown project") from error


@app.post("/api/knowledge/upload")
async def upload_documents(
    project_id: str = Form(...),
    files: list[UploadFile] = File(...),
    scope: str = Form("public"),
) -> list[dict]:
    if scope not in {"public", "private"}:
        raise HTTPException(status_code=422, detail="scope must be public or private")
    tasks = []
    try:
        for upload in files:
            tasks.append(
                _knowledge().enqueue(project_id, upload.filename or "document", await upload.read(), scope=scope)
            )
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Unknown project") from error
    except ValueError as error:
        raise HTTPException(status_code=415, detail=str(error)) from error
    return tasks


class CollectPaperRequest(BaseModel):
    project_id: str
    paper: Paper


@app.post("/api/knowledge/collect")
def collect_paper(request: CollectPaperRequest) -> dict:
    try:
        return _knowledge().collect_paper(request.project_id, request.paper)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Unknown project") from error


@app.get("/api/projects/{project_id}/documents")
def list_documents(project_id: str) -> list[dict]:
    try:
        return _knowledge().documents(project_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Unknown project") from error


@app.delete("/api/knowledge/documents/{document_id}", status_code=204)
def delete_document(document_id: str) -> None:
    try:
        _knowledge().delete_document(document_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Unknown document") from error


class ConversationRequest(BaseModel):
    project_id: str
    title: str = Field(default="New conversation", max_length=120)


@app.post("/api/conversations")
def create_conversation(request: ConversationRequest) -> dict:
    try:
        return _knowledge().create_conversation(request.project_id, request.title)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Unknown project") from error


@app.get("/api/projects/{project_id}/conversations")
def list_conversations(project_id: str) -> list[dict]:
    try:
        return _knowledge().conversations(project_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Unknown project") from error


@app.get("/api/conversations/{conversation_id}")
def get_conversation(conversation_id: str) -> dict:
    try:
        return _knowledge().conversation(conversation_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Unknown conversation") from error


@app.delete("/api/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str) -> None:
    try:
        _knowledge().delete_conversation(conversation_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Unknown conversation") from error


@app.get("/api/knowledge/tasks/{document_id}")
def document_task(document_id: str) -> dict:
    try:
        return _knowledge().document(document_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Unknown document") from error


@app.get("/api/projects/{project_id}/export")
def export_project(project_id: str) -> JSONResponse:
    try:
        return JSONResponse(_knowledge().export(project_id), headers={"Content-Disposition": f'attachment; filename="{project_id}.json"'})
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Unknown project") from error


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    corpus: str | None = None
    source: str = Field(default="arxiv", pattern="^(arxiv|local|project)$")
    project_id: str | None = None
    locale: Literal["zh", "en"] = "zh"
    ranking: Literal["auto", "relevance", "recent", "citations"] = "auto"
    constraints: ResearchConstraints = Field(default_factory=ResearchConstraints)
    paper_limit: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def infer_paper_limit(self) -> "AskRequest":
        import re
        match = re.search(r"(?:调查|查找|分析|阅读)?\s*(\d{1,2})\s*(?:篇|papers?)", self.question, re.I)
        if match:
            self.paper_limit = max(1, min(20, int(match.group(1))))
        return self


class ChatRequest(BaseModel):
    messages: list[ConversationMessage] = Field(min_length=1, max_length=20)
    locale: Literal["zh", "en"] = "zh"
    project_id: str | None = None
    conversation_id: str | None = None


@app.post("/api/chat", response_model=ConversationResult)
def chat(request: ChatRequest) -> ConversationResult:
    try:
        conversation_id = request.conversation_id
        if request.project_id and not conversation_id:
            conversation_id = _knowledge().create_conversation(
                request.project_id, request.messages[0].content
            )["id"]
        settings = get_settings()
        memory = (
            [entry["content"] for entry in _knowledge().list_memory(request.project_id)]
            if request.project_id
            else []
        )
        result = understand_request(request.messages, settings, request.locale, memory=memory)
        if result.status == "ready" and request.project_id:
            try:
                project_corpus = _knowledge().corpus_path(request.project_id)
            except KeyError:
                project_corpus = None
            if project_corpus and Path(project_corpus).exists():
                with CorpusStore(Path(project_corpus)) as store:
                    if store.paper_count() > 0:
                        answer, citations = answer_with_project_sources(
                            result.refined_question or request.messages[-1].content,
                            store,
                            settings,
                            request.locale,
                        )
                        if answer:
                            result.message = answer
                        result.citations = citations
        if conversation_id:
            persisted = [message.model_dump() for message in request.messages]
            persisted.append({"role": "assistant", "content": result.message})
            _knowledge().save_messages(conversation_id, persisted)
            result.conversation_id = conversation_id
        if request.project_id and settings.use_model_reasoning:
            for item in remember_conversation(request.messages, settings):
                _knowledge().add_memory(request.project_id, item)
        return result
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Unknown project or conversation") from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/projects/{project_id}/memory")
def list_project_memory(project_id: str) -> list[dict]:
    try:
        return _knowledge().list_memory(project_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Unknown project") from error


@app.delete("/api/projects/{project_id}/memory/{memory_id}", status_code=204)
def delete_project_memory(project_id: str, memory_id: int) -> None:
    try:
        _knowledge().delete_memory(project_id, memory_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Unknown project or memory entry") from error


def _ranking_for(request: AskRequest) -> str:
    if request.ranking != "auto":
        return request.ranking
    text = request.question.casefold()
    if any(term in text for term in ("最新", "近期", "recent", "latest", "newest")):
        return "recent"
    if any(term in text for term in ("高引用", "经典", "影响力", "citation", "influential")):
        return "citations"
    return "relevance"


def _corpus_path(request: AskRequest, settings: Settings) -> Path:
    if request.source == "project":
        if not request.project_id:
            raise HTTPException(status_code=422, detail="Select a project knowledge base")
        try:
            return _knowledge().corpus_path(request.project_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Unknown project") from error
    default_path = Path(settings.data_dir) / "corpus.sqlite"
    corpus_path = Path(request.corpus) if request.corpus else default_path
    resolved = corpus_path.resolve()
    if Path(settings.data_dir).resolve() not in resolved.parents:
        raise HTTPException(status_code=422, detail="Corpus path must live under the data directory")
    return resolved


def _require_populated_corpus(corpus_path: Path) -> None:
    if not corpus_path.exists():
        raise HTTPException(status_code=404, detail=f"Corpus does not exist: {corpus_path}")
    with CorpusStore(corpus_path) as store:
        if store.paper_count() == 0:
            raise HTTPException(status_code=422, detail="Corpus contains no papers")


def _prepare_source(
    request: AskRequest, settings: Settings
) -> tuple[Path, Settings, str | None, int, str | None]:
    source_warning = None
    if request.source == "project" and not request.corpus:
        corpus_path = _corpus_path(request, settings)
        if corpus_path.exists():
            with CorpusStore(corpus_path) as store:
                if store.paper_count() > 0:
                    return corpus_path, settings.model_copy(update={"max_papers": request.paper_limit}), None, 0, None
        source_warning = "当前项目知识库为空，已自动切换为 arXiv 检索。"
        request = request.model_copy(update={"source": "arxiv"})
    if request.source == "local" or request.corpus:
        corpus_path = _corpus_path(request, settings)
        _require_populated_corpus(corpus_path)
        return corpus_path, settings.model_copy(update={"max_papers": request.paper_limit}), None, 0, None
    try:
        search_query, inferred_ranking = interpret_query(request.question, settings)
        ranking = _ranking_for(request)
        if request.ranking == "auto" and inferred_ranking:
            ranking = inferred_ranking
        documents = search_arxiv(
            request.question,
            max_results=min(request.paper_limit, settings.arxiv_max_results, 20),
            timeout_seconds=min(settings.arxiv_timeout_seconds, 15.0),
            api_url=settings.arxiv_api_url,
            cache_dir=Path(settings.data_dir) / "arxiv-cache",
            max_retries=1,
            ranking=ranking,
            search_query=search_query,
        )
        if not documents:
            raise ArxivSearchError("arXiv returned no matching papers")
        query_key = hashlib.sha256(
            f"v4\0{request.question.strip().casefold()}\0{search_query}\0{ranking}".encode()
        ).hexdigest()[:16]
        corpus_path = Path(settings.data_dir) / "arxiv-queries" / f"{query_key}.sqlite"
        with CorpusStore(corpus_path) as store:
            for document in documents:
                store.upsert(document)
        _prune_arxiv_cache(Path(settings.data_dir) / "arxiv-queries")
        source_settings = settings.model_copy(
            update={"retrieval_mode": "lexical", "max_papers": request.paper_limit}
        )
        return corpus_path, source_settings, source_warning, len(documents), search_query
    except ArxivSearchError as error:
        raise ArxivSearchError(
            f"arXiv 暂时不可用，未返回本地 SciFact 结果以避免混入无关论文：{error}"
        ) from error


def _prune_arxiv_cache(cache_dir: Path, keep: int = 100, max_age_days: int = 14) -> None:
    """Cap arXiv per-query corpora: drop files older than max_age_days, keep newest `keep`."""
    if not cache_dir.exists():
        return
    import time

    files = sorted(cache_dir.glob("*.sqlite"), key=lambda p: p.stat().st_mtime, reverse=True)
    cutoff = time.time() - max_age_days * 86400
    for index, path in enumerate(files):
        if index >= keep or path.stat().st_mtime < cutoff:
            path.unlink(missing_ok=True)



@app.get("/health")
def health() -> dict[str, object]:
    settings = get_settings()
    corpus_path = Path(settings.data_dir) / "corpus.sqlite"
    paper_count = 0
    if corpus_path.exists():
        with CorpusStore(corpus_path) as store:
            paper_count = store.paper_count()
    return {"status": "ok", "corpus": str(corpus_path), "paper_count": paper_count}


@app.post("/api/ask")
def ask(request: AskRequest) -> dict[str, object]:
    settings = get_settings()
    try:
        corpus_path, agent_settings, source_warning, _, search_query = _prepare_source(request, settings)
    except ArxivSearchError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    with CorpusStore(corpus_path) as store:
        state = GrantScoutAgent(
            store,
            agent_settings,
            decompose=request.source != "arxiv",
            output_language=request.locale,
        ).run(
            request.question,
            search_query=search_query,
            constraints=request.constraints,
        )
    if source_warning:
        state.warnings.insert(0, source_warning)
    payload = state.model_dump(mode="json")
    payload["report"] = render_markdown(state)
    payload["report_html"] = render_html(state, request.locale)
    payload["report_fragment"] = render_html_fragment(state, request.locale)
    payload["report_fragments"] = {
        "zh": render_html_fragment(state, "zh"),
        "en": render_html_fragment(state, "en"),
    }
    if request.project_id:
        try:
            payload["archived_report"] = _knowledge().save_report(
                request.project_id, request.question, payload["report"]
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Unknown project") from error
    return payload


@app.post("/api/ask/stream")
def ask_stream(request: AskRequest) -> StreamingResponse:
    """Stream tool outcomes and the final traceable report as NDJSON."""
    settings = get_settings()
    if request.source == "local" or request.corpus:
        _require_populated_corpus(_corpus_path(request, settings))

    def stream() -> Iterator[str]:
        events: Queue[dict[str, Any] | None] = Queue()

        def publish(event_type: str, **payload: Any) -> None:
            events.put({"type": event_type, **payload})

        def worker() -> None:
            try:
                publish("started", question=request.question)
                publish("source_started", source=request.source)
                corpus_path, agent_settings, source_warning, imported, search_query = _prepare_source(
                    request, settings
                )
                publish(
                    "source_completed",
                    source="local" if source_warning else request.source,
                    imported=imported,
                    warning=source_warning,
                )
                with CorpusStore(corpus_path) as store:
                    agent = GrantScoutAgent(
                        store,
                        agent_settings,
                        decompose=request.source != "arxiv",
                        output_language=request.locale,
                        on_tool_call=lambda tool_call: publish(
                            "tool_call", tool_call=tool_call.model_dump(mode="json")
                        ),
                    )
                    state = agent.run(
                        request.question,
                        search_query=search_query,
                        constraints=request.constraints,
                    )
                if source_warning:
                    state.warnings.insert(0, source_warning)
                report_markdown = render_markdown(state)
                archived_report = None
                if request.project_id:
                    archived_report = _knowledge().save_report(
                        request.project_id, request.question, report_markdown
                    )
                publish(
                    "completed",
                    state=state.model_dump(mode="json"),
                    report=report_markdown,
                    report_html=render_html(state, request.locale),
                    report_fragment=render_html_fragment(state, request.locale),
                    report_fragments={
                        "zh": render_html_fragment(state, "zh"),
                        "en": render_html_fragment(state, "en"),
                    },
                    archived_report=archived_report,
                )
            except Exception as error:
                publish("error", message=str(error))
            finally:
                events.put(None)

        Thread(target=worker, name="grantscout-review", daemon=True).start()
        while True:
            event = events.get()
            if event is None:
                break
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/proposals/draft")
def draft_proposal(request: ProposalRequest) -> dict[str, object]:
    """Draft a full proposal from the local corpus behind the data directory."""
    settings = get_settings()
    corpus_path = _resolve_corpus(request.corpus, settings)
    if not corpus_path.exists():
        raise HTTPException(status_code=404, detail=f"Corpus does not exist: {corpus_path}")
    with CorpusStore(corpus_path) as store:
        if store.paper_count() == 0:
            raise HTTPException(status_code=422, detail="Corpus contains no papers")
        try:
            document = ProposalPipeline(
                store,
                settings,
                template_id=request.template_id,
                output_language=request.locale,
            ).run(request)
        except ProposalPipelineError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
    _proposals().save(document)
    return {
        "document": document.model_dump(mode="json"),
        "report": render_proposal_markdown(document),
    }


def _resolve_corpus(corpus: str | None, settings: Settings) -> Path:
    """Only corpora inside the configured data directory may be opened."""
    data_root = Path(settings.data_dir).resolve()
    corpus_path = (
        (data_root / "corpus.sqlite") if not corpus else Path(corpus)
    ).resolve()
    if data_root not in corpus_path.parents:
        raise HTTPException(status_code=422, detail="Corpus path must live under the data directory")
    return corpus_path


def _data_relative(file_path: str, settings: Settings) -> Path:
    """Resolve a viewer/files request path; reject anything outside data dir."""
    data_root = Path(settings.data_dir).resolve()
    target = (data_root / file_path).resolve()
    if target != data_root and data_root not in target.parents:
        raise HTTPException(status_code=422, detail="Path escapes the data directory")
    return target


@app.get("/files/{file_path:path}")
def serve_data_file(file_path: str) -> FileResponse:
    """Serve library/knowledge files for the built-in citation viewer."""
    settings = get_settings()
    target = _data_relative(file_path, settings)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(target)


@app.get("/viewer")
def file_viewer(file: str, page: int = 1) -> HTMLResponse:
    """Built-in viewer page for citation jump links (native PDF #page anchor)."""
    settings = get_settings()
    target = _data_relative(file, settings)
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if target.suffix.lower() not in {".pdf", ".txt", ".md"}:
        raise HTTPException(status_code=415, detail="Unsupported preview format")
    from urllib.parse import quote

    src = f"/files/{quote(str(target.relative_to(Path(settings.data_dir).resolve())))}"
    if target.suffix.lower() == ".pdf":
        src += f"#page={max(page, 1)}"
    return HTMLResponse(
        f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>原文查看 - {file}</title>
<style>html,body{{margin:0;height:100%;background:#172033}}embed,iframe{{width:100%;height:96vh;border:0}}
.bar{{background:#0f766e;color:#fff;font:14px system-ui;padding:8px 14px}}
a{{color:#fff}}</style></head><body>
<div class="bar">原文查看:{file}{f" · 第 {page} 页" if target.suffix.lower() == ".pdf" else ""} ·
<a href="{src}" target="_blank">新窗口打开</a></div>
<embed src="{src}" type="application/pdf">
</body></html>"""
    )


def _load_proposal(proposal_id: str) -> "object":
    try:
        return _proposals().get(proposal_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Unknown proposal") from error


@app.post("/api/completion")
def inline_completion(request: CompletionRequest) -> dict[str, object]:
    settings = get_settings()
    corpus_path = _resolve_corpus(request.corpus, settings)
    if not corpus_path.exists():
        raise HTTPException(status_code=404, detail=f"Corpus does not exist: {corpus_path}")
    if not hasattr(app.state, "completion_cache"):
        app.state.completion_cache = CompletionCache()
    cache: CompletionCache = app.state.completion_cache
    cache.register(request.session_id, request.request_id)
    semantic_index = None
    if settings.retrieval_mode.lower() == "semantic":
        from grantscout.retrieval.semantic import SemanticIndex

        semantic_index = SemanticIndex(settings.vector_index_path, settings.embedding_model)
        try:
            semantic_index.load()
        except Exception:  # noqa: BLE001 - semantic is optional
            semantic_index = None
    with CorpusStore(corpus_path) as store:
        engine = CompletionEngine(store, settings, semantic_index=semantic_index, cache=cache)
        response = engine.run_command(request) if request.command else engine.complete(request)
    return response.model_dump(mode="json")


class PaperMetaRequest(BaseModel):
    corpus: str | None = None
    paper_id: str
    scope: Literal["public", "private"] | None = None
    venue: str | None = None
    language: str | None = None
    tags: list[str] | None = None
    cited_by_count: int | None = Field(default=None, ge=0)


@app.patch("/api/corpus/paper-meta")
def set_paper_meta(request: PaperMetaRequest) -> dict:
    settings = get_settings()
    corpus_path = _resolve_corpus(request.corpus, settings)
    if not corpus_path.exists():
        raise HTTPException(status_code=404, detail=f"Corpus does not exist: {corpus_path}")
    with CorpusStore(corpus_path) as store:
        try:
            store.set_paper_meta(
                request.paper_id,
                scope=request.scope,
                venue=request.venue,
                language=request.language,
                tags=request.tags,
                cited_by_count=request.cited_by_count,
            )
            return store.get_paper_meta(request.paper_id)
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/api/evidence/{evidence_id}")
def get_evidence(evidence_id: str, corpus: str | None = None) -> dict:
    settings = get_settings()
    corpus_path = _resolve_corpus(corpus, settings)
    if not corpus_path.exists():
        raise HTTPException(status_code=404, detail=f"Corpus does not exist: {corpus_path}")
    with CorpusStore(corpus_path) as store:
        evidence = store.get_evidence(evidence_id)
        if evidence is None:
            raise HTTPException(status_code=404, detail="Unknown evidence")
        paper = store.get_paper(evidence.paper_id)
        return {
            "evidence": evidence.model_dump(mode="json"),
            "paper_title": paper.title if paper else evidence.paper_id,
            "scope": (store.get_paper_meta(evidence.paper_id).get("scope") or "public"),
        }




class ArxivCiteRequest(BaseModel):
    query: str = Field(min_length=2, max_length=300)
    max_results: int = Field(default=5, ge=1, le=20)


@app.post("/api/arxiv/cite")
def arxiv_cite(request: ArxivCiteRequest) -> list[dict]:
    """arXiv 检索并返回引用格式;网页解析直接输出,不经过 LLM。"""
    settings = get_settings()
    try:
        documents = search_arxiv(
            request.query,
            max_results=min(request.max_results, settings.arxiv_max_results),
            timeout_seconds=min(settings.arxiv_timeout_seconds, 15.0),
            api_url=settings.arxiv_api_url,
        )
    except ArxivSearchError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return [
        {
            "title": doc.paper.title,
            "bibtex": to_bibtex(doc.paper),
            "gbt7714": to_gbt7714(doc.paper),
        }
        for doc in documents
    ]


@app.get("/playground", response_class=HTMLResponse)
def completion_playground() -> str:
    return Path(__file__).with_name("playground.html").read_text(encoding="utf-8")


@app.get("/api/proposals")
def list_proposals() -> list[dict]:
    return _proposals().list()


@app.get("/api/proposals/{proposal_id}")
def get_proposal(proposal_id: str) -> dict[str, object]:
    document = _load_proposal(proposal_id)
    return {
        "document": document.model_dump(mode="json"),
        "report": render_proposal_markdown(document),
    }


class OutlineApproval(BaseModel):
    approved: bool


@app.post("/api/proposals/{proposal_id}/outline/approve")
def approve_outline(proposal_id: str, request: OutlineApproval) -> dict[str, object]:
    document = _load_proposal(proposal_id)
    if document.outline is None:
        raise HTTPException(status_code=422, detail="Proposal has no outline yet")
    document.outline.approved = request.approved
    _proposals().save(document)
    return {"document": document.model_dump(mode="json")}


@app.post("/api/proposals/{proposal_id}/sections/{section_key}/regenerate")
def regenerate_proposal_section(proposal_id: str, section_key: str) -> dict[str, object]:
    document = _load_proposal(proposal_id)
    settings = get_settings()
    corpus_path = _resolve_corpus(document.corpus_path, settings)
    if not corpus_path.exists():
        raise HTTPException(status_code=404, detail=f"Corpus does not exist: {corpus_path}")
    with CorpusStore(corpus_path) as store:
        pipeline = ProposalPipeline(
            store,
            settings,
            template_id=document.template_id,
            output_language=document.locale,
        )
        try:
            document = pipeline.redraft_section(document, section_key)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
    return {
        "document": document.model_dump(mode="json"),
        "report": render_proposal_markdown(document),
    }


@app.get("/api/proposals/{proposal_id}/report.md")
def proposal_report_markdown(proposal_id: str) -> PlainTextResponse:
    document = _load_proposal(proposal_id)
    return PlainTextResponse(
        render_proposal_markdown(document),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{proposal_id}.md"'},
    )


@app.get("/api/proposals/{proposal_id}/report.docx")
def proposal_report_docx(proposal_id: str) -> Response:
    document = _load_proposal(proposal_id)
    runs_dir = Path(get_settings().runs_dir)
    docx_path = runs_dir / f"{document.id}.proposal.docx"
    if not docx_path.exists():
        raise HTTPException(status_code=404, detail="DOCX export unavailable for this proposal")
    return Response(
        content=docx_path.read_bytes(),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{proposal_id}.docx"'},
    )


@app.get("/proposals", response_class=HTMLResponse)
def proposal_workbench() -> str:
    return Path(__file__).with_name("proposal.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return Path(__file__).with_name("index.html").read_text(encoding="utf-8")
