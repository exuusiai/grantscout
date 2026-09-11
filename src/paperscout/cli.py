import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer

from paperscout.agent.loop import PaperScoutAgent
from paperscout.config import Settings, get_settings
from paperscout.evaluation.datasets import ingest_jsonl
from paperscout.evaluation.ablation import run_ablation, run_comparison
from paperscout.evaluation.runner import run_evaluation
from paperscout.evaluation.qasper import persist_qasper_evaluation, run_qasper_answer_evaluation
from paperscout.evaluation.suite import persist_evaluation_suite, run_evaluation_suite
from paperscout.logging import configure_logging
from paperscout.models.llm import ModelClientError, OpenAICompatibleClient
from paperscout.retrieval.parser import parse_document
from paperscout.retrieval.semantic import SemanticIndex, SemanticIndexError
from paperscout.retrieval.reranker import CrossEncoderReranker, RerankerError
from paperscout.retrieval.store import CorpusStore
from paperscout.reports.renderer import render_markdown
from paperscout.tools.search import search_papers

app = typer.Typer(no_args_is_help=True, add_completion=False)
logger = logging.getLogger(__name__)


def _client(settings: Settings) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        base_url=settings.model_base_url,
        api_key=settings.model_api_key,
        model=settings.model_name,
        timeout_seconds=settings.model_timeout_seconds,
    )


def _emit(payload: dict[str, Any]) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@app.command()
def health() -> None:
    """Check local configuration and the configured model endpoint."""
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.prepare_directories()
    model_status = _client(settings).healthcheck()
    _emit(
        {
            "status": "ok" if model_status["ok"] else "degraded",
            "timestamp": datetime.now(UTC).isoformat(),
            "environment": settings.environment,
            "storage": {
                "data_dir": str(settings.data_dir),
                "runs_dir": str(settings.runs_dir),
            },
            "model": {
                "base_url": settings.model_base_url,
                "name": settings.model_name,
                "reachable": model_status["ok"],
                "details": model_status.get("models", model_status.get("error")),
            },
        }
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Research question to process."),
    dry_run: bool = typer.Option(False, help="Validate the request without calling a model."),
    corpus: Path = typer.Option(Path("data/corpus.sqlite"), help="SQLite corpus path."),
) -> None:
    """Send a research question to the configured model and return JSON."""
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.prepare_directories()
    started_at = datetime.now(UTC)
    result: dict[str, Any] = {
        "question": question,
        "status": "planned" if dry_run else "started",
        "started_at": started_at.isoformat(),
        "trajectory": [],
    }
    result["trajectory"].append(
        {
            "step": 0,
            "action": "receive_question",
            "input": {"question": question},
            "status": "ok",
        }
    )
    if dry_run:
        result["message"] = "Phase 0 dry run validated; model call skipped."
        result["finished_at"] = datetime.now(UTC).isoformat()
        _emit(result)
        return

    if corpus.exists():
        with CorpusStore(corpus) as store:
            if store.paper_count() > 0:
                state = PaperScoutAgent(store, settings).run(question)
                payload = state.model_dump(mode="json")
                payload["report"] = render_markdown(state)
                _emit(payload)
                return

    prompt = (
        "You are the PaperScout research assistant. Return a concise JSON object with "
        "keys: interpretation, sub_questions, caveats. Do not invent citations.\n\n"
        f"Research question: {question}"
    )
    try:
        response = _client(settings).chat(
            messages=[
                {"role": "system", "content": "You are a careful scientific research assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=settings.model_max_tokens,
            temperature=settings.model_temperature,
        )
        result.update(
            {
                "status": "completed",
                "model": response.model or settings.model_name,
                "answer": response.content,
                "usage": response.usage,
            }
        )
        result["trajectory"].append(
            {"step": 1, "action": "model_chat", "status": "ok", "usage": response.usage}
        )
    except ModelClientError as error:
        logger.error("Model request failed: %s", error)
        result.update(
            {
                "status": "failed",
                "error": {"type": "model_client", "message": str(error), "retryable": True},
            }
        )
        result["trajectory"].append(
            {
                "step": 1,
                "action": "model_chat",
                "status": "error",
                "error": str(error),
            }
        )
    result["finished_at"] = datetime.now(UTC).isoformat()
    _emit(result)


@app.command()
def ingest(
    path: Path = typer.Argument(..., exists=True, readable=True, help="TXT, Markdown, or PDF paper."),
    paper_id: str | None = typer.Option(None, help="Stable paper identifier."),
    title: str | None = typer.Option(None, help="Paper title override."),
    year: int | None = typer.Option(None, min=0, max=3000, help="Publication year."),
    corpus: Path = typer.Option(Path("data/corpus.sqlite"), help="SQLite corpus path."),
) -> None:
    """Parse a paper and add its sections and evidence to the local corpus."""
    settings = get_settings()
    configure_logging(settings.log_level)
    document = parse_document(path, paper_id=paper_id, title=title, year=year)
    with CorpusStore(corpus) as store:
        store.upsert(document)
        _emit(
            {
                "status": "ingested",
                "paper": document.paper.model_dump(mode="json"),
                "sections": len(document.sections),
                "evidence_items": len(document.evidence_items),
                "corpus": str(corpus),
            }
        )


@app.command(name="ingest-jsonl")
def ingest_jsonl_command(
    path: Path = typer.Argument(..., exists=True, readable=True, help="Qasper, SciFact, or generic JSONL file."),
    corpus: Path = typer.Option(Path("data/corpus.sqlite"), help="SQLite corpus path."),
) -> None:
    """Import dataset-style JSONL records into the local corpus."""
    with CorpusStore(corpus) as store:
        counts = ingest_jsonl(path, store)
    _emit({"status": "ingested", "source": str(path), "corpus": str(corpus), **counts})


@app.command(name="search")
def search_command(
    query: str = typer.Argument(..., help="Search query."),
    top_k: int = typer.Option(10, min=1, max=100, help="Maximum results."),
    corpus: Path = typer.Option(Path("data/corpus.sqlite"), help="SQLite corpus path."),
    mode: str = typer.Option("lexical", help="Retrieval mode: lexical or semantic."),
    rerank: bool = typer.Option(False, help="Use the optional Cross-Encoder reranker."),
) -> None:
    """Search local evidence and return explainable results."""
    if not corpus.exists():
        _emit({"status": "error", "error": f"Corpus does not exist: {corpus}"})
        raise typer.Exit(code=1)
    with CorpusStore(corpus) as store:
        settings = get_settings()
        semantic_index = (
            SemanticIndex(settings.vector_index_path, settings.embedding_model) if mode == "semantic" else None
        )
        reranker = CrossEncoderReranker(settings.reranker_model) if rerank else None
        try:
            candidates = search_papers(
                store, query, top_k=top_k, semantic_index=semantic_index, reranker=reranker
            )
        except (SemanticIndexError, RerankerError) as error:
            _emit({"status": "error", "error": str(error)})
            raise typer.Exit(code=1) from error
        _emit(
            {
                "status": "ok",
                "query": query,
                "results": [candidate.model_dump(mode="json") for candidate in candidates],
            }
        )


@app.command()
def index(
    corpus: Path = typer.Option(Path("data/corpus.sqlite"), help="SQLite corpus path."),
    output: Path | None = typer.Option(None, help="Vector index path."),
    model: str | None = typer.Option(None, help="Sentence-transformer model name."),
) -> None:
    """Build a persistent semantic index for local evidence."""
    settings = get_settings()
    index_path = output or settings.vector_index_path
    model_name = model or settings.embedding_model
    try:
        with CorpusStore(corpus) as store:
            result = SemanticIndex(index_path, model_name).build(store)
    except SemanticIndexError as error:
        _emit({"status": "error", "error": str(error)})
        raise typer.Exit(code=1) from error
    _emit({"status": "indexed", **result})


@app.command()
def evaluate(
    queries: Path = typer.Argument(..., exists=True, readable=True, help="Evaluation JSONL file."),
    corpus: Path = typer.Option(Path("data/corpus.sqlite"), help="SQLite corpus path."),
    top_k: int = typer.Option(10, min=1, max=100, help="Retrieval cutoff."),
    mode: str = typer.Option("lexical", help="Retrieval mode: lexical or semantic."),
) -> None:
    """Evaluate local retrieval with Recall@K, evidence recall, and MRR."""
    if not corpus.exists():
        _emit({"status": "error", "error": f"Corpus does not exist: {corpus}"})
        raise typer.Exit(code=1)
    settings = get_settings()
    try:
        result = run_evaluation(corpus, queries, top_k=top_k, mode=mode, settings=settings)
    except (ValueError, SemanticIndexError) as error:
        _emit({"status": "error", "error": str(error)})
        raise typer.Exit(code=1) from error
    _emit({"status": "ok", **result})


@app.command(name="evaluate-qasper-answers")
def evaluate_qasper_answers(
    queries: Path = typer.Argument(..., exists=True, readable=True, help="Qasper evaluation JSONL."),
    corpus: Path = typer.Option(Path("data/qasper.sqlite"), help="Qasper SQLite corpus."),
    output: Path = typer.Option(
        Path("runs/qasper-answer-evaluation.json"), help="Prediction and score artifact."
    ),
    top_k: int = typer.Option(10, min=1, max=50, help="Evidence retrieval cutoff."),
    generation_top_k: int | None = typer.Option(
        None, min=1, max=50, help="Optional number of retrieved passages sent to the model."
    ),
    mode: str = typer.Option("semantic", help="Retrieval mode: lexical or semantic."),
    limit: int | None = typer.Option(None, min=1, help="Optional smoke-test query limit."),
    workers: int = typer.Option(4, min=1, max=32, help="Concurrent local-model requests."),
    checkpoint: Path = typer.Option(
        Path("runs/qasper-answer-predictions.jsonl"), help="Resume-safe prediction checkpoint."
    ),
) -> None:
    """Generate local-model Qasper answers and score all official answer types."""
    if not corpus.exists():
        _emit({"status": "error", "error": f"Corpus does not exist: {corpus}"})
        raise typer.Exit(code=1)
    settings = get_settings()
    try:
        payload = run_qasper_answer_evaluation(
            corpus,
            queries,
            settings,
            top_k=top_k,
            mode=mode,
            limit=limit,
            workers=workers,
            checkpoint_path=checkpoint,
            generation_top_k=generation_top_k,
        )
    except (ValueError, SemanticIndexError, json.JSONDecodeError) as error:
        _emit({"status": "error", "error": str(error)})
        raise typer.Exit(code=1) from error
    artifacts = persist_qasper_evaluation(payload, output)
    _emit({"status": "ok", "artifacts": artifacts, **payload})


@app.command()
def benchmark(
    queries: Path = typer.Argument(..., exists=True, readable=True, help="Evaluation JSONL file."),
    corpus: Path = typer.Option(Path("data/corpus.sqlite"), help="SQLite corpus path."),
    top_k: int = typer.Option(5, min=1, max=100, help="Retrieval cutoff."),
) -> None:
    """Compare fixed RAG with PaperScout Agentic RAG."""
    if not corpus.exists():
        _emit({"status": "error", "error": f"Corpus does not exist: {corpus}"})
        raise typer.Exit(code=1)
    _emit({"status": "ok", **run_comparison(corpus, queries, get_settings(), top_k=top_k)})


@app.command()
def ablate(
    queries: Path = typer.Argument(..., exists=True, readable=True, help="Evaluation JSONL file."),
    corpus: Path = typer.Option(Path("data/corpus.sqlite"), help="SQLite corpus path."),
    top_k: int = typer.Option(5, min=1, max=100, help="Retrieval cutoff."),
) -> None:
    """Run planning, reranking, audit, and conflict-detection ablations."""
    if not corpus.exists():
        _emit({"status": "error", "error": f"Corpus does not exist: {corpus}"})
        raise typer.Exit(code=1)
    _emit({"status": "ok", **run_ablation(corpus, queries, get_settings(), top_k=top_k)})


@app.command(name="evaluate-suite")
def evaluate_suite(
    queries: Path = typer.Argument(..., exists=True, readable=True, help="Evaluation JSONL file."),
    corpus: Path = typer.Option(Path("data/corpus.sqlite"), help="SQLite corpus path."),
    top_k: int = typer.Option(5, min=1, max=100, help="Retrieval cutoff."),
    output_dir: Path | None = typer.Option(None, help="Directory for suite JSON and Markdown artifacts."),
) -> None:
    """Run retrieval, benchmark, and ablation evaluation and persist one result bundle."""
    if not corpus.exists():
        _emit({"status": "error", "error": f"Corpus does not exist: {corpus}"})
        raise typer.Exit(code=1)
    settings = get_settings()
    settings.prepare_directories()
    payload = run_evaluation_suite(corpus, queries, settings, top_k=top_k)
    artifacts = persist_evaluation_suite(payload, output_dir or settings.runs_dir)
    _emit({"status": "ok", "artifacts": artifacts, **payload})


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8000, min=1, max=65535, help="Bind port."),
) -> None:
    """Start the minimal PaperScout web UI and API."""
    import uvicorn

    uvicorn.run("paperscout.api.app:app", host=host, port=port, reload=False)
