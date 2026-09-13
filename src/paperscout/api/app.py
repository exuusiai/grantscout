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

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PaperScout</title>
<style>
:root{color:#172033;background:#f7fafc;font-family:system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}*{box-sizing:border-box}body{margin:0}main{max-width:1440px;margin:auto;padding:24px}header{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #cbd5e1;padding-bottom:16px}h1{font-size:24px;margin:0}h2{font-size:16px;margin:0 0 12px}h3{font-size:14px;margin:0}.header-actions{display:flex;align-items:center;gap:12px}.status{color:#475569;font-size:14px}.question{display:grid;grid-template-columns:1fr auto;gap:12px;padding:20px 0;border-bottom:1px solid #cbd5e1}textarea{width:100%;min-height:92px;resize:vertical;border:1px solid #94a3b8;border-radius:4px;padding:12px;font:inherit;color:inherit;background:#fff}button{align-self:end;border:1px solid #0f766e;background:#0f766e;color:#fff;border-radius:4px;padding:10px 16px;font:inherit;cursor:pointer;min-width:112px}button:disabled{background:#94a3b8;border-color:#94a3b8;cursor:wait}.language{min-width:40px;padding:6px 9px;background:#fff;color:#0f766e}.workspace{display:grid;grid-template-columns:minmax(230px,.7fr) minmax(280px,1fr) minmax(360px,1.7fr);gap:24px;padding-top:24px}.panel{min-width:0}.panel+ .panel{border-left:1px solid #cbd5e1;padding-left:24px}.timeline,.list{list-style:none;margin:0;padding:0}.timeline li,.list li{padding:10px 0;border-bottom:1px solid #e2e8f0;font-size:14px}.timeline .error{color:#b91c1c}.timeline time{display:block;color:#64748b;font-size:12px;margin-bottom:3px}.empty{color:#64748b;font-size:14px}.report{min-height:560px;white-space:pre-wrap;overflow-wrap:anywhere;margin:0;padding:14px;background:#fff;border:1px solid #cbd5e1;border-radius:4px;font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}.audit{margin-top:20px;border-top:1px solid #cbd5e1;padding-top:16px}.warning{color:#9a3412}.running{color:#0f766e;font-weight:600}.hidden{display:none}@media(max-width:980px){.workspace{grid-template-columns:1fr 1fr}.report-panel{grid-column:1/-1}.panel+ .panel{border-left:0;padding-left:0}}@media(max-width:640px){main{padding:16px}.question{grid-template-columns:1fr}.workspace{grid-template-columns:1fr;gap:20px}.report-panel{grid-column:auto}.panel+ .panel{border-top:1px solid #cbd5e1;padding-top:20px}.report{min-height:360px}}
</style></head><body><main>
<header><h1>PaperScout 论文侦察</h1><div class="header-actions"><span id="status" class="status">就绪 / Ready</span><button id="language" class="language" type="button" title="切换语言 / Switch language">EN</button></div></header>
<section class="question"><textarea id="question" aria-label="研究问题 / Research question" placeholder="输入研究问题，例如：查找 GRPO 相关论文"></textarea><button id="run" type="button">开始综述</button></section>
<section class="workspace">
<section class="panel"><h2 id="timeline-title">运行进度 / Timeline</h2><ol id="timeline" class="timeline"><li class="empty">尚未开始任务。</li></ol></section>
<section class="panel"><h2 id="evidence-title">选定证据 / Evidence</h2><ul id="evidence" class="list"><li class="empty">尚未选定证据。</li></ul><section class="audit"><h2 id="audit-title">引用审计 / Citation Audit</h2><div id="audit" class="empty">尚未完成审计。</div></section><section class="audit"><h2 id="warnings-title">提示 / Warnings</h2><ul id="warnings" class="list"><li class="empty">暂无提示。</li></ul></section></section>
<section class="panel report-panel"><h2 id="report-title">最终报告 / Final Report</h2><pre id="report" class="report">开始综述后，这里将显示可追溯报告。</pre></section>
</section></main><script>
const question=document.querySelector('#question'),run=document.querySelector('#run'),language=document.querySelector('#language'),status=document.querySelector('#status'),timeline=document.querySelector('#timeline'),evidence=document.querySelector('#evidence'),audit=document.querySelector('#audit'),warnings=document.querySelector('#warnings'),report=document.querySelector('#report');let locale='zh',terminalEvent=false;
const copy={zh:{ready:'就绪 / Ready',run:'开始综述',running:'正在分析，请稍候…',starting:'正在连接分析服务…',emptyTimeline:'尚未开始任务。',emptyEvidence:'尚未选定证据。',emptyWarnings:'暂无提示。',waitingAudit:'等待引用审计…',disabledAudit:'引用审计未启用。',noEvidence:'本地语料库中没有选出匹配证据。',failed:'运行失败',completed:'已完成',question:'请输入至少 3 个字符的研究问题。',closed:'分析连接提前结束，请重试。',report:'开始综述后，这里将显示可追溯报告。'},en:{ready:'Ready',run:'Run review',running:'Analyzing…',starting:'Connecting…',emptyTimeline:'No run started.',emptyEvidence:'No evidence selected.',emptyWarnings:'No warnings.',waitingAudit:'Waiting for citation audit…',disabledAudit:'Citation audit disabled.',noEvidence:'No matching evidence was selected from the local corpus.',failed:'Failed',completed:'Completed',question:'Enter a research question of at least 3 characters.',closed:'The analysis stream ended early. Please retry.',report:'Run a review to generate a traceable report.'}};const t=key=>copy[locale][key];
function clear(node){node.textContent=''}function item(node,text,kind=''){const li=document.createElement('li');li.textContent=text;if(kind)li.className=kind;node.append(li)}function formatTime(value){return value?new Date(value).toLocaleTimeString():'now'}
function addTool(call){if(timeline.querySelector('.empty'))clear(timeline);const li=document.createElement('li');li.className=call.status==='error'?'error':'';const time=document.createElement('time');time.textContent=formatTime(call.timestamp);const title=document.createElement('strong');title.textContent=call.tool;const detail=document.createElement('div');detail.textContent=JSON.stringify(call.input);li.append(time,title,detail);timeline.append(li)}
function renderState(state){clear(evidence);const papers=new Map(state.selected_papers.map(p=>[p.id,p.title]));if(state.evidence_items.length){for(const itemData of state.evidence_items){item(evidence,(papers.get(itemData.paper_id)||itemData.paper_id)+' · '+itemData.id)}}else{item(evidence,t('noEvidence'),'empty')}const citation=state.citation_audit;if(citation){audit.textContent=citation.status+' · '+citation.supported_claims+' supported · '+citation.unsupported_claims+' needs review'}else{audit.textContent=t('disabledAudit')}clear(warnings);if(state.warnings.length){for(const warning of state.warnings)item(warnings,warning,'warning')}else{item(warnings,t('emptyWarnings'),'empty')}}
function handleEvent(event){if(event.type==='started'){status.textContent=t('running');status.className='status running'}if(event.type==='tool_call')addTool(event.tool_call);if(event.type==='completed'){terminalEvent=true;status.textContent=t('completed');status.className='status';renderState(event.state);report.textContent=event.report}if(event.type==='error'){terminalEvent=true;status.textContent=t('failed');status.className='status';item(timeline,event.message,'error')}}
async function runReview(){const value=question.value.trim();if(value.length<3){status.textContent=t('question');question.focus();return}terminalEvent=false;run.disabled=true;run.textContent=t('running');status.textContent=t('starting');status.className='status running';clear(timeline);item(timeline,t('starting'),'running');clear(evidence);item(evidence,t('running'),'empty');clear(warnings);item(warnings,t('emptyWarnings'),'empty');audit.textContent=t('waitingAudit');report.textContent=t('running');try{const response=await fetch('/api/ask/stream',{method:'POST',cache:'no-store',headers:{'Content-Type':'application/json','Accept':'application/x-ndjson'},body:JSON.stringify({question:value})});if(!response.ok){let message='Request failed';try{message=(await response.json()).detail||message}catch{}throw new Error(message)}if(!response.body)throw new Error('Streaming response is unavailable');const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='';while(true){const chunk=await reader.read();buffer+=decoder.decode(chunk.value||new Uint8Array(),{stream:!chunk.done});const lines=buffer.split('\n');buffer=lines.pop()||'';for(const line of lines)if(line.trim())handleEvent(JSON.parse(line));if(chunk.done)break}if(buffer.trim())handleEvent(JSON.parse(buffer));if(!terminalEvent)throw new Error(t('closed'))}catch(error){status.textContent=t('failed');status.className='status';item(timeline,error instanceof Error?error.message:String(error),'error')}finally{run.disabled=false;run.textContent=t('run')}}
function setLocale(next){locale=next;document.documentElement.lang=locale==='zh'?'zh-CN':'en';language.textContent=locale==='zh'?'EN':'中';run.textContent=t('run');question.placeholder=locale==='zh'?'输入研究问题，例如：查找 GRPO 相关论文':'Ask a research question, for example: Find papers about GRPO';if(!run.disabled)status.textContent=t('ready')}
language.addEventListener('click',()=>setLocale(locale==='zh'?'en':'zh'));run.addEventListener('click',runReview);question.addEventListener('keydown',event=>{if((event.metaKey||event.ctrlKey)&&event.key==='Enter')runReview()});
</script></body></html>"""
