from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from paperscout.agent.loop import PaperScoutAgent
from paperscout.config import Settings, get_settings
from paperscout.reports.renderer import render_html, render_markdown
from paperscout.retrieval.store import CorpusStore

app = FastAPI(title="PaperScout", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    corpus: str | None = None


def _corpus_path(request: AskRequest, settings: Settings) -> Path:
    default_path = Path(settings.data_dir) / "corpus.sqlite"
    return Path(request.corpus) if request.corpus else default_path


def _require_populated_corpus(corpus_path: Path) -> None:
    if not corpus_path.exists():
        raise HTTPException(status_code=404, detail=f"Corpus does not exist: {corpus_path}")
    with CorpusStore(corpus_path) as store:
        if store.paper_count() == 0:
            raise HTTPException(status_code=422, detail="Corpus contains no papers")


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
    corpus_path = _corpus_path(request, settings)
    _require_populated_corpus(corpus_path)
    with CorpusStore(corpus_path) as store:
        state = PaperScoutAgent(store, settings).run(request.question)
    payload = state.model_dump(mode="json")
    payload["report"] = render_markdown(state)
    payload["report_html"] = render_html(state)
    return payload


@app.post("/api/ask/stream")
def ask_stream(request: AskRequest) -> StreamingResponse:
    """Stream tool outcomes and the final traceable report as NDJSON."""
    settings = get_settings()
    corpus_path = _corpus_path(request, settings)
    _require_populated_corpus(corpus_path)

    def stream() -> Iterator[str]:
        events: Queue[dict[str, Any] | None] = Queue()

        def publish(event_type: str, **payload: Any) -> None:
            events.put({"type": event_type, **payload})

        def worker() -> None:
            try:
                publish("started", question=request.question)
                with CorpusStore(corpus_path) as store:
                    agent = PaperScoutAgent(
                        store,
                        settings,
                        on_tool_call=lambda tool_call: publish(
                            "tool_call", tool_call=tool_call.model_dump(mode="json")
                        ),
                    )
                    state = agent.run(request.question)
                publish(
                    "completed",
                    state=state.model_dump(mode="json"),
                    report=render_markdown(state),
                    report_html=render_html(state),
                )
            except Exception as error:
                publish("error", message=str(error))
            finally:
                events.put(None)

        Thread(target=worker, name="paperscout-review", daemon=True).start()
        while True:
            event = events.get()
            if event is None:
                break
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PaperScout</title>
<style>
:root{color:#172033;background:#f7fafc;font-family:system-ui,-apple-system,sans-serif}*{box-sizing:border-box}body{margin:0}main{max-width:1440px;margin:auto;padding:24px}header{display:flex;justify-content:space-between;align-items:baseline;border-bottom:1px solid #cbd5e1;padding-bottom:16px}h1{font-size:24px;margin:0}h2{font-size:16px;margin:0 0 12px}h3{font-size:14px;margin:0}.status{color:#475569;font-size:14px}.question{display:grid;grid-template-columns:1fr auto;gap:12px;padding:20px 0;border-bottom:1px solid #cbd5e1}textarea{width:100%;min-height:92px;resize:vertical;border:1px solid #94a3b8;border-radius:4px;padding:12px;font:inherit;color:inherit;background:#fff}button{align-self:end;border:1px solid #0f766e;background:#0f766e;color:#fff;border-radius:4px;padding:10px 16px;font:inherit;cursor:pointer}button:disabled{background:#94a3b8;border-color:#94a3b8;cursor:wait}.workspace{display:grid;grid-template-columns:minmax(230px,.7fr) minmax(280px,1fr) minmax(360px,1.7fr);gap:24px;padding-top:24px}.panel{min-width:0}.panel+ .panel{border-left:1px solid #cbd5e1;padding-left:24px}.timeline,.list{list-style:none;margin:0;padding:0}.timeline li,.list li{padding:10px 0;border-bottom:1px solid #e2e8f0;font-size:14px}.timeline .error{color:#b91c1c}.timeline time{display:block;color:#64748b;font-size:12px;margin-bottom:3px}.label{color:#475569;font-size:12px;text-transform:uppercase}.empty{color:#64748b;font-size:14px}.report{min-height:560px;white-space:pre-wrap;overflow-wrap:anywhere;margin:0;padding:14px;background:#fff;border:1px solid #cbd5e1;border-radius:4px;font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}.audit{margin-top:20px;border-top:1px solid #cbd5e1;padding-top:16px}.warning{color:#9a3412}.hidden{display:none}@media(max-width:980px){.workspace{grid-template-columns:1fr 1fr}.report-panel{grid-column:1/-1}.panel+ .panel{border-left:0;padding-left:0}}@media(max-width:640px){main{padding:16px}.question{grid-template-columns:1fr}.workspace{grid-template-columns:1fr;gap:20px}.report-panel{grid-column:auto}.panel+ .panel{border-top:1px solid #cbd5e1;padding-top:20px}.report{min-height:360px}}
</style></head><body><main>
<header><h1>PaperScout</h1><span id="status" class="status">Ready</span></header>
<section class="question"><textarea id="question" aria-label="Research question" placeholder="Ask a research question"></textarea><button id="run" type="button">Run review</button></section>
<section class="workspace">
<section class="panel"><h2>Run Timeline</h2><ol id="timeline" class="timeline"><li class="empty">No run started.</li></ol></section>
<section class="panel"><h2>Selected Evidence</h2><ul id="evidence" class="list"><li class="empty">No evidence selected.</li></ul><section class="audit"><h2>Citation Audit</h2><div id="audit" class="empty">No audit completed.</div></section><section class="audit"><h2>Warnings</h2><ul id="warnings" class="list"><li class="empty">No warnings.</li></ul></section></section>
<section class="panel report-panel"><h2>Final Report</h2><pre id="report" class="report">Run a review to generate a traceable report.</pre></section>
</section></main><script>
const question=document.querySelector('#question'),run=document.querySelector('#run'),status=document.querySelector('#status'),timeline=document.querySelector('#timeline'),evidence=document.querySelector('#evidence'),audit=document.querySelector('#audit'),warnings=document.querySelector('#warnings'),report=document.querySelector('#report');
function clear(node){node.textContent=''}function item(node,text,kind=''){const li=document.createElement('li');li.textContent=text;if(kind)li.className=kind;node.append(li)}function formatTime(value){return value?new Date(value).toLocaleTimeString():'now'}
function addTool(call){if(timeline.querySelector('.empty'))clear(timeline);const li=document.createElement('li');li.className=call.status==='error'?'error':'';const time=document.createElement('time');time.textContent=formatTime(call.timestamp);const title=document.createElement('strong');title.textContent=call.tool;const detail=document.createElement('div');detail.textContent=JSON.stringify(call.input);li.append(time,title,detail);timeline.append(li)}
function renderState(state){clear(evidence);const papers=new Map(state.selected_papers.map(p=>[p.id,p.title]));if(state.evidence_items.length){for(const itemData of state.evidence_items){item(evidence,(papers.get(itemData.paper_id)||itemData.paper_id)+' · '+itemData.id)}}else{item(evidence,'No evidence selected.','empty')}const citation=state.citation_audit;if(citation){audit.textContent=citation.status+' · '+citation.supported_claims+' supported · '+citation.unsupported_claims+' needs review'}else{audit.textContent='Citation audit disabled.'}clear(warnings);if(state.warnings.length){for(const warning of state.warnings)item(warnings,warning,'warning')}else{item(warnings,'No warnings.','empty')}}
function handleEvent(event){if(event.type==='started'){status.textContent='Running'}if(event.type==='tool_call')addTool(event.tool_call);if(event.type==='completed'){status.textContent='Completed';renderState(event.state);report.textContent=event.report;run.disabled=false}if(event.type==='error'){status.textContent='Failed';item(timeline,event.message,'error');run.disabled=false}}
async function runReview(){const value=question.value.trim();if(value.length<3){status.textContent='Enter a research question.';return}run.disabled=true;status.textContent='Starting';clear(timeline);clear(evidence);clear(warnings);audit.textContent='Waiting for citation audit.';report.textContent='';try{const response=await fetch('/api/ask/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:value})});if(!response.ok)throw new Error((await response.json()).detail||'Request failed');const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='';while(true){const {value,done}=await reader.read();buffer+=decoder.decode(value||new Uint8Array(),{stream:!done});const lines=buffer.split('\n');buffer=lines.pop();for(const line of lines)if(line)handleEvent(JSON.parse(line));if(done)break}if(buffer)handleEvent(JSON.parse(buffer))}catch(error){status.textContent='Failed';item(timeline,error.message,'error');run.disabled=false}}
run.addEventListener('click',runReview);question.addEventListener('keydown',event=>{if((event.metaKey||event.ctrlKey)&&event.key==='Enter')runReview()});
</script></body></html>"""
