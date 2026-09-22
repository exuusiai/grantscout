"""Inline completion engine: L0 cache, L1 filters, L2 lexical, L3 semantic, L5 LLM.

Latency budget is 1-3s; verbatim (Type C) and lexical (Type B) paths are
LLM-free. Every candidate carries its evidence citations. The public/private
scope hard-gate lands with the metadata layer (Phase 2); until then requests
are served from the corpus they name.
"""

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

from grantscout.completion.filters import FilterSet, ParsedPrefix, parse_filters
from grantscout.completion.schemas import (
    CompletionCandidate,
    CompletionCitation,
    CompletionRequest,
    CompletionResponse,
)
from grantscout.config import Settings
from grantscout.models.llm import ModelClientError, OpenAICompatibleClient, strip_thinking
from grantscout.models.schemas import EvidenceItem
from grantscout.retrieval.semantic import SemanticIndex
from grantscout.retrieval.sentences import SentenceIndex, split_sentences
from grantscout.retrieval.store import CorpusStore

_MARKER_PATTERN = re.compile(r"【文献(\d{1,2})】")
_CACHE_TTL_SECONDS = 120.0
_CACHE_MAX_ENTRIES = 512
_KIND_SCORE = {"verbatim": 1.0, "lexical": 0.8, "semantic": 0.7, "llm": 0.5, "command": 0.6}


class CompletionCache:
    """L0: short-TTL response cache plus best-effort stale-request tracking."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[float, CompletionResponse]] = {}
        self._superseded: set[str] = set()
        self._latest: dict[str, str] = {}
        self._lock = threading.Lock()

    @staticmethod
    def key(request: CompletionRequest, parsed: ParsedPrefix) -> str:
        filters = parsed.filters
        filter_signature = json.dumps(
            {
                "filename": sorted(filters.filename),
                "author": sorted(filters.author),
                "tags": sorted(filters.tags),
                "conferences": sorted(filters.conferences),
                "impact_min": filters.impact_min,
                "time": [filters.time_from, filters.time_to],
            },
            sort_keys=True,
        )
        material = "\0".join(
            [
                request.corpus or "",
                parsed.clean_text,
                filter_signature,
                request.suffix,
                request.locale,
                request.mode,
                str(request.command or ""),
                request.command_text,
                str(request.max_candidates),
            ]
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def get(self, key: str) -> CompletionResponse | None:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            stored_at, response = entry
            if now - stored_at > _CACHE_TTL_SECONDS:
                self._entries.pop(key, None)
                return None
            return response.model_copy(deep=True, update={"cache_hit": True})

    def put(self, key: str, response: CompletionResponse) -> None:
        with self._lock:
            if len(self._entries) >= _CACHE_MAX_ENTRIES:
                oldest = min(self._entries, key=lambda k: self._entries[k][0])
                self._entries.pop(oldest, None)
            self._entries[key] = (time.monotonic(), response.model_copy(deep=True))

    def register(self, session_id: str | None, request_id: str | None) -> None:
        if not session_id or not request_id:
            return
        with self._lock:
            previous = self._latest.get(session_id)
            if previous and previous != request_id:
                self._superseded.add(previous)
                if len(self._superseded) > 1024:
                    self._superseded.clear()
            self._latest[session_id] = request_id

    def is_superseded(self, session_id: str | None, request_id: str | None) -> bool:
        if not session_id or not request_id:
            return False
        with self._lock:
            return request_id in self._superseded


@dataclass
class _Context:
    parsed: ParsedPrefix
    paper_ids: set[str] | None = None


class CompletionEngine:
    def __init__(
        self,
        store: CorpusStore,
        settings: Settings,
        semantic_index: SemanticIndex | None = None,
        cache: CompletionCache | None = None,
    ) -> None:
        self.store = store
        self.settings = settings
        self.semantic_index = semantic_index
        self.cache = cache or CompletionCache()
        self.sentences = SentenceIndex(store)
        self._scope_map: dict[str, str] | None = None
        self.model_client = self._build_model_client(settings)

    @staticmethod
    def _build_model_client(settings: Settings) -> OpenAICompatibleClient | None:
        if not settings.use_model_reasoning:
            return None
        return OpenAICompatibleClient(
            base_url=settings.model_base_url,
            api_key=settings.model_api_key,
            model=settings.model_name,
            timeout_seconds=min(settings.model_timeout_seconds, 30.0),
        )

    # Public entry points ---------------------------------------------------
    def complete(self, request: CompletionRequest) -> CompletionResponse:
        started = time.perf_counter()
        parsed = parse_filters(request.prefix)
        context = _Context(parsed=parsed, paper_ids=self._paper_allowlist(parsed.filters))
        key = self.cache.key(request, parsed)
        cached = self.cache.get(key)
        if cached is not None:
            cached.superseded = self.cache.is_superseded(request.session_id, request.request_id)
            cached.latency_ms = int((time.perf_counter() - started) * 1000)
            return cached
        response = self._complete_uncached(request, context)
        response.latency_ms = int((time.perf_counter() - started) * 1000)
        response.superseded = self.cache.is_superseded(request.session_id, request.request_id)
        self.cache.put(key, response)
        return response

    def run_command(self, request: CompletionRequest) -> CompletionResponse:
        started = time.perf_counter()
        text = request.command_text.strip() or request.prefix.strip()
        if request.command == "find":
            response = self._find_command(request)
        elif request.command == "summarize":
            response = self._summarize_command(request, text)
        elif request.command == "refine":
            response = self._refine_command(request, text)
        else:  # pragma: no cover - schema constrains values
            response = CompletionResponse()
        response.latency_ms = int((time.perf_counter() - started) * 1000)
        return response

    # Pipeline stages --------------------------------------------------------
    def _complete_uncached(self, request: CompletionRequest, context: _Context) -> CompletionResponse:
        response = CompletionResponse(
            filters_applied=_filters_applied(context.parsed.filters),
            filter_warnings=list(context.parsed.warnings),
        )
        mode = request.mode
        if mode in ("auto", "verbatim"):
            response.candidates.extend(self._verbatim_candidates(request, context))
        if mode in ("auto", "lexical"):
            response.candidates.extend(self._lexical_candidates(request, context))
        if mode in ("auto", "semantic") and self.semantic_index is not None:
            response.candidates.extend(self._semantic_candidates(request, context))
        if mode in ("auto", "llm") and self.model_client is not None:
            response.candidates.extend(self._llm_candidates(request, context, response))
        response.candidates = self._select(response.candidates, request.max_candidates)
        return response

    def _verbatim_candidates(self, request: CompletionRequest, context: _Context) -> list[CompletionCandidate]:
        prefix = context.parsed.clean_text
        if not prefix.strip():
            return []
        hits = self.sentences.continue_fragment(prefix, top_k=2, paper_ids=context.paper_ids)
        candidates = []
        for hit in hits:
            candidates.append(
                CompletionCandidate(
                    text=hit.remainder,
                    kind="verbatim",
                    score=_KIND_SCORE["verbatim"],
                    trigger="照抄补全(库内原文)",
                    citations=[
                        CompletionCitation(
                            evidence_id=hit.evidence.id,
                            paper_id=hit.evidence.paper_id,
                            paper_title=self._paper_title(hit.evidence.paper_id),
                            page=hit.evidence.page,
                            scope=self._scope_of(hit.evidence.paper_id),
                            **self._citation_meta(hit.evidence),
                        )
                    ],
                )
            )
        return candidates

    def _lexical_candidates(self, request: CompletionRequest, context: _Context) -> list[CompletionCandidate]:
        query = _query_terms(context.parsed.clean_text)
        if not query:
            return []
        results = self.store.search(query, top_k=4, paper_ids=context.paper_ids)
        return self._evidence_to_candidates(results, kind="lexical")

    def _semantic_candidates(self, request: CompletionRequest, context: _Context) -> list[CompletionCandidate]:
        query = context.parsed.clean_text.strip()
        if not query:
            return []
        try:
            results = self.semantic_index.search(query, top_k=4, paper_ids=context.paper_ids)
        except Exception as error:  # noqa: BLE001 - optional dependency
            del error
            return []
        return self._evidence_to_candidates(results, kind="semantic")

    def _llm_candidates(
        self, request: CompletionRequest, context: _Context, response: CompletionResponse
    ) -> list[CompletionCandidate]:
        query = _query_terms(context.parsed.clean_text)
        sources = self._retrieve_sources(query, context.paper_ids) if query else []
        try:
            self._ensure_local_model_allowed(sources)
        except ModelClientError as error:
            response.filter_warnings.append(str(error))
            return []
        try:
            content = self.model_client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an inline writing assistant for a research proposal. "
                            "Continue the user's text at <CURSOR> with 1-2 sentences"
                            + (" in Chinese" if request.locale == "zh" else " in English")
                            + ". Use only the supplied sources; end any sentence that uses a "
                            "source with its 【文献n】 marker; never invent facts or citations. "
                            "Output ONLY the continuation text, no headings, no explanations."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"{request.prefix}<CURSOR>{request.suffix}\n\n"
                            + "\n".join(
                                f"【文献{index}】 {item.text}" for index, item in enumerate(sources, start=1)
                            )
                        ),
                    },
                ],
                max_tokens=256,
                temperature=0.2,
                chat_template_kwargs={"enable_thinking": False},
            ).content
        except ModelClientError as error:
            response.filter_warnings.append(f"模型补全不可用,仅返回检索类建议:{error}")
            return []
        content = strip_thinking(content).strip()
        if not content:
            return []
        citations = []
        known_titles = {item.id: self._paper_title(item.paper_id) for item in sources}

        def replace(match: re.Match[str]) -> str:
            index = int(match.group(1))
            if not 1 <= index <= len(sources):
                return ""
            evidence = sources[index - 1]
            citations.append(
                CompletionCitation(
                    evidence_id=evidence.id,
                    paper_id=evidence.paper_id,
                    paper_title=known_titles[evidence.id],
                    page=evidence.page,
                    scope=self._scope_of(evidence.paper_id),
                    **self._citation_meta(evidence),
                )
            )
            return match.group(0)

        cleaned = _MARKER_PATTERN.sub(replace, content).strip()
        if not cleaned:
            return []
        return [
            CompletionCandidate(
                text=cleaned,
                kind="llm",
                score=_KIND_SCORE["llm"],
                trigger="模型续写",
                citations=citations,
            )
        ]

    def _retrieve_sources(self, query: str, paper_ids: set[str] | None, top_k: int = 3) -> list[EvidenceItem]:
        """Lexical first; fall back to the trigram sentence index for CJK text.

        The main FTS table tokenizes CJK runs as whole tokens, so short
        Chinese prefixes often miss lexically; the trigram index matches
        substrings and keeps Type B working for Chinese.
        """
        results = self.store.search(query, top_k=top_k, paper_ids=paper_ids)
        if results:
            return [result.evidence for result in results]
        hits = self.sentences.continue_fragment(query, top_k=top_k, paper_ids=paper_ids)
        return [hit.evidence for hit in hits]

    def _evidence_to_candidates(self, results, kind: str) -> list[CompletionCandidate]:
        candidates: list[CompletionCandidate] = []
        seen_sentences: set[str] = set()
        for result in results:
            sentence = _best_sentence(result.evidence.text, result.matched_terms)
            if not sentence or sentence in seen_sentences:
                continue
            seen_sentences.add(sentence)
            candidates.append(
                CompletionCandidate(
                    text=sentence,
                    kind=kind,  # type: ignore[arg-type]
                    score=_KIND_SCORE[kind],
                    trigger=f"库内检索({kind})",
                    citations=[
                        CompletionCitation(
                            evidence_id=result.evidence.id,
                            paper_id=result.evidence.paper_id,
                            paper_title=result.paper.title,
                            page=result.evidence.page,
                            scope=self._scope_of(result.evidence.paper_id),
                            **self._citation_meta(result.evidence),
                        )
                    ],
                )
            )
        return candidates

    # Commands ----------------------------------------------------------------
    def _find_command(self, request: CompletionRequest) -> CompletionResponse:
        parsed = parse_filters(request.command_text or request.prefix)
        context = _Context(parsed=parsed, paper_ids=self._paper_allowlist(parsed.filters))
        response = CompletionResponse(
            filters_applied=_filters_applied(parsed.filters),
            filter_warnings=list(parsed.warnings),
        )
        response.candidates.extend(self._lexical_candidates(request, context))
        if self.semantic_index is not None:
            response.candidates.extend(self._semantic_candidates(request, context))
        response.candidates = self._select(response.candidates, request.max_candidates)
        return response

    def _summarize_command(self, request: CompletionRequest, text: str) -> CompletionResponse:
        if not text:
            return CompletionResponse(filter_warnings=["没有可总结的文本。"])
        if self.model_client is not None:
            try:
                content = self.model_client.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Summarize the supplied text in "
                                + ("2-3 句中文" if request.locale == "zh" else "2-3 English sentences")
                                + ". Do not add facts that are absent from the text. Output the summary only."
                            ),
                        },
                        {"role": "user", "content": text},
                    ],
                    max_tokens=384,
                    temperature=0.1,
                    chat_template_kwargs={"enable_thinking": False},
                ).content
                return CompletionResponse(
                    candidates=[
                        CompletionCandidate(
                            text=strip_thinking(content).strip(), kind="command", score=0.6, trigger="/summarize"
                        )
                    ]
                )
            except ModelClientError as error:
                return CompletionResponse(filter_warnings=[f"模型总结失败,降级为抽取式摘要:{error}"])
        sentences = [s for s in split_sentences(text) if s[2] - s[1] > 10][:2]
        summary = "".join(s[0] for s in sentences) or text[:200]
        return CompletionResponse(
            candidates=[
                CompletionCandidate(
                    text=summary, kind="command", score=0.6, trigger="/summarize(抽取式降级)"
                )
            ],
            filter_warnings=["模型未启用,使用抽取式摘要。"],
        )

    def _refine_command(self, request: CompletionRequest, text: str) -> CompletionResponse:
        if not text:
            return CompletionResponse(filter_warnings=["没有可润色的文本。"])
        if self.model_client is None:
            return CompletionResponse(filter_warnings=["模型未启用,无法润色,返回原文。"])
        try:
            content = self.model_client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an academic writing editor. Polish the supplied passage for a "
                            "research proposal without changing its meaning, facts, or citations. "
                            + ("Output polished Chinese only." if request.locale == "zh" else "Output polished English only.")
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=768,
                temperature=0.2,
                chat_template_kwargs={"enable_thinking": False},
            ).content
            return CompletionResponse(
                candidates=[
                    CompletionCandidate(
                        text=strip_thinking(content).strip(), kind="command", score=0.6, trigger="/refine"
                    )
                ]
            )
        except ModelClientError as error:
            return CompletionResponse(filter_warnings=[f"润色失败:{error}"])

    # Helpers -------------------------------------------------------------------
    def _paper_allowlist(self, filters: FilterSet) -> set[str] | None:
        sets: list[set[str]] = []
        connection = self.store.connection
        for name in filters.filename:
            pattern = f"%{name}%"
            rows = connection.execute(
                "SELECT id FROM papers WHERE title LIKE ? OR source_path LIKE ?", (pattern, pattern)
            ).fetchall()
            sets.append({row[0] for row in rows} or {""})
        for author in filters.author:
            rows = connection.execute(
                "SELECT id FROM papers WHERE authors_json LIKE ?", (f"%{author}%",)
            ).fetchall()
            sets.append({row[0] for row in rows} or {""})
        for tag in filters.tags:
            rows = connection.execute(
                "SELECT id FROM papers WHERE tags LIKE ?", (f"%{tag}%",)
            ).fetchall()
            sets.append({row[0] for row in rows} or {""})
        for venue in filters.conferences:
            rows = connection.execute(
                "SELECT id FROM papers WHERE venue LIKE ?", (f"%{venue}%",)
            ).fetchall()
            sets.append({row[0] for row in rows} or {""})
        if filters.impact_min is not None:
            rows = connection.execute(
                "SELECT id FROM papers WHERE cited_by_count >= ?", (filters.impact_min,)
            ).fetchall()
            sets.append({row[0] for row in rows} or {""})
        if filters.time_from is not None or filters.time_to is not None:
            conditions, parameters = [], []
            if filters.time_from is not None:
                conditions.append("year >= ?")
                parameters.append(filters.time_from)
            if filters.time_to is not None:
                conditions.append("year <= ?")
                parameters.append(filters.time_to)
            rows = connection.execute(
                f"SELECT id FROM papers WHERE {' AND '.join(conditions)}", parameters
            ).fetchall()
            sets.append({row[0] for row in rows} or {""})
        if not sets:
            return None
        allowlist = set.intersection(*sets)
        return allowlist or {""}

    def _scope_of(self, paper_id: str) -> str:
        if self._scope_map is None:
            rows = self.store.connection.execute("SELECT id, scope FROM papers").fetchall()
            self._scope_map = {row[0]: (row[1] or "public") for row in rows}
        return self._scope_map.get(paper_id, "public")

    def _ensure_local_model_allowed(self, sources: list[EvidenceItem]) -> None:
        """Hard privacy gate: private evidence must never reach a remote model."""
        if not any(self._scope_of(item.paper_id) == "private" for item in sources):
            return
        base_url = getattr(self.model_client, "base_url", "") if self.model_client else ""
        host = urlsplit(base_url).hostname
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ModelClientError("私域证据不允许调用外部模型接口,仅允许本地回环端点。")

    def _paper_title(self, paper_id: str) -> str:
        paper = self.store.get_paper(paper_id)
        return paper.title if paper else paper_id

    def _citation_meta(self, evidence: EvidenceItem) -> dict:
        """source_path plus a jump URL: built-in /viewer by default, external pdf.js when configured."""
        paper = self.store.get_paper(evidence.paper_id)
        source_path = paper.source_path if paper else None
        url = None
        base = self.settings.file_server_base_url.strip()
        if base and source_path:
            from urllib.parse import quote

            url = f"{base.rstrip('/')}/viewer.html?file={quote(source_path)}"
            if evidence.page:
                url += f"#page={evidence.page}"
        elif source_path:
            from pathlib import Path

            from urllib.parse import quote

            data_root = Path(self.settings.data_dir).resolve()
            target = Path(source_path)
            attempts = [target] if target.is_absolute() else [data_root / target, Path.cwd() / target]
            relative = None
            for candidate in attempts:
                try:
                    relative = str(candidate.resolve().relative_to(data_root))
                    break
                except ValueError:
                    continue
            if relative is not None:
                url = f"/viewer?file={quote(relative)}"
                if evidence.page:
                    url += f"&page={evidence.page}"
        return {"source_path": source_path, "source_url": url}

    @staticmethod
    def _select(candidates: list[CompletionCandidate], limit: int) -> list[CompletionCandidate]:
        selected: list[CompletionCandidate] = []
        seen: set[str] = set()
        by_kind: dict[str, list[CompletionCandidate]] = {}
        for candidate in candidates:
            if candidate.text in seen:
                continue
            seen.add(candidate.text)
            by_kind.setdefault(candidate.kind, []).append(candidate)
        order = ["verbatim", "lexical", "semantic", "llm", "command"]
        while len(selected) < limit:
            added = False
            for kind in order:
                queue = by_kind.get(kind)
                if queue:
                    selected.append(queue.pop(0))
                    added = True
                    if len(selected) >= limit:
                        break
            if not added:
                break
        return selected


def _filters_applied(filters: FilterSet) -> dict[str, list[str]]:
    applied: dict[str, list[str]] = {}
    if filters.filename:
        applied["filename"] = list(filters.filename)
    if filters.author:
        applied["author"] = list(filters.author)
    if filters.tags:
        applied["tag"] = list(filters.tags)
    if filters.conferences:
        applied["conference"] = list(filters.conferences)
    if filters.impact_min is not None:
        applied["impact"] = [f">={filters.impact_min}"]
    if filters.time_from is not None or filters.time_to is not None:
        applied["time"] = [f"{filters.time_from or ''}-{filters.time_to or ''}"]
    return applied


def _query_terms(text: str) -> str:
    """Keep the last sentence or so of the prefix as the retrieval query."""
    tail = text.strip()[-160:]
    return tail.strip()


def _best_sentence(text: str, matched_terms: list[str]) -> str:
    sentences = [item[0] for item in split_sentences(text)]
    if not sentences:
        return text.strip()[:200]
    if not matched_terms:
        return sentences[0]
    lowered = [term.lower() for term in matched_terms]

    def overlap(sentence: str) -> int:
        lower = sentence.lower()
        return sum(1 for term in lowered if term in lower)

    return max(sentences, key=overlap)
