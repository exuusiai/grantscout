# PaperScout

PaperScout is an evidence-driven literature review agent for scientific papers.
It keeps a traceable relationship between question, paper, evidence, claim, and
citation audit instead of returning an unsupported summary.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
paperscout health
paperscout ask "What methods reduce hallucinations in RAG?" --dry-run
paperscout ingest papers/example.pdf --corpus data/corpus.sqlite
paperscout search "retrieval hallucination" --corpus data/corpus.sqlite
paperscout ask "What methods reduce hallucinations in RAG?" --corpus data/corpus.sqlite
./examples/run_demo.sh
```

The model client targets an OpenAI-compatible API. Set `PAPERSCOUT_MODEL_BASE_URL`,
`PAPERSCOUT_MODEL_NAME`, and `PAPERSCOUT_MODEL_API_KEY` in `.env` when a local
inference server is available.

The model URL is validated before any network request: only loopback hosts are
accepted, redirects are disabled, and third-party gateways or `/responses`
paths are rejected.

## Full workflow

- Import TXT, Markdown, PDF, Qasper-style JSON, and SciFact-style JSONL.
- Build a SQLite corpus with stable section and evidence IDs.
- Use lexical retrieval by default, or BGE-M3 plus FAISS after installing
  `pip install -e '.[retrieval]'` and running `paperscout index`.
- Run deterministic PaperScout Agentic RAG with budgets, warnings, comparisons,
  citation audit, and JSONL trajectory persistence.
- Evaluate Recall@K, evidence Recall@K, MRR, fixed RAG, and ablations.
- Evaluate Qasper answer F1, answer type accuracy, yes/no, unanswerable,
  extractive/free-form answers, evidence F1, latency, and token usage.
- Start a minimal Web UI with `paperscout serve --host 0.0.0.0 --port 8000`.

## Data and model setup

Download source files with recorded hashes:

```bash
python scripts/download_datasets.py --dataset scifact --split all
python scripts/download_datasets.py --dataset qasper --split train
paperscout ingest-jsonl data/raw/scifact/corpus.jsonl --corpus data/corpus.sqlite
```

The downloader uses the official Qasper and SciFact archives, extracts the
requested files into `data/raw/{dataset}`, and records portable relative paths
and SHA256 hashes in `manifest.json`.

Build evaluation query files from the official annotations after importing the
matching corpus:

```bash
python scripts/build_scifact_eval.py \
  --claims data/raw/scifact/claims_dev.jsonl \
  --corpus-jsonl data/raw/scifact/corpus.jsonl \
  --corpus data/scifact.sqlite \
  --output evals/scifact/dev.jsonl
python scripts/build_qasper_eval.py \
  --source data/raw/qasper/dev.json \
  --corpus data/qasper.sqlite \
  --output evals/qasper/dev.jsonl
```

Run resume-safe Qasper answer generation and scoring against the local model:

```bash
PAPERSCOUT_VECTOR_INDEX_PATH=data/qasper.bge-small-en-v1.5.auto.index \
paperscout evaluate-qasper-answers evals/qasper/dev.jsonl \
  --corpus data/qasper.sqlite --mode semantic --top-k 10 --workers 4 \
  --checkpoint runs/qasper-answer-predictions.jsonl \
  --output runs/qasper-answer-evaluation.json
```

Lexical retrieval uses SQLite FTS5 with automatic index backfill. Environments
without FTS5 retain the deterministic scan fallback. Evaluation suites reuse a
loaded semantic model/index across benchmark and ablation stages.

Qasper answer evaluation defaults to `top-k=10` because the paper-constrained
semantic retrieval audit found evidence recall of 0.6653 at K=10 versus 0.4849
at K=5. Use K=5 for a lower-context baseline and report the cutoff with every
answer-quality result.

For semantic retrieval:

```bash
python -m pip install -e '.[retrieval]'
paperscout index --corpus data/corpus.sqlite
PAPERSCOUT_RETRIEVAL_MODE=semantic paperscout ask "..." --corpus data/corpus.sqlite
```

Semantic retrieval automatically falls back to CPU when the installed PyTorch
build does not support the detected GPU architecture. Set
`PAPERSCOUT_EMBEDDING_DEVICE=cpu` to force CPU explicitly.

For local generation, install the optional model dependencies and start
`scripts/start_vllm.sh` on port `8001`; the PaperScout Web UI uses port `8000`.
The deterministic path remains available if the model endpoint is down.

## Local model preflight

PaperScout intentionally permits only local model endpoints. It never sends a
model request to a third-party OpenAI-compatible gateway or to `/v1/responses`.
Before starting vLLM, validate the model directory and serving environment:

```bash
# Verify downloaded weights without requiring a CUDA/vLLM installation.
python3 scripts/preflight_vllm.py --model-only --model-path /path/to/Qwen-Qwen3-8B

# Run this in the CUDA Python environment that has vLLM installed.
PAPERSCOUT_PYTHON=/path/to/python ./scripts/start_vllm.sh

# After vLLM is running, probe only the local endpoint.
PAPERSCOUT_PYTHON=/path/to/python scripts/preflight_vllm.py \
  --model-path /path/to/Qwen-Qwen3-8B \
  --check-endpoint http://127.0.0.1:8001/v1
paperscout health
```

The launcher disables the FlashInfer sampler by default because the installed
FlashInfer build can reject RTX 5090 `sm_120` during JIT setup. Override with
`PAPERSCOUT_USE_FLASHINFER_SAMPLER=1` only after validating the installed build.

The preflight verifies every shard listed in `model.safetensors.index.json`,
checks vLLM and CUDA/Torch compatibility, and rejects a non-loopback endpoint.
The vLLM launcher also binds only to a loopback address by default and rejects
an external `PAPERSCOUT_MODEL_HOST` override.
For an RTX 5090, use a Torch build that includes `sm_120`; otherwise the
preflight fails before vLLM starts.

See `docs/architecture.md`, `docs/evaluation.md`, and `docs/security.md` for the
design and reproducibility contract.

The current SciFact dev retrieval comparison is recorded in
`docs/scifact-retrieval-comparison.md`. It contains separate lexical and
semantic runs; the semantic run uses the local BGE index and no external model
gateway.

Known reproducible failure cases and their handling are documented in
`docs/failure_cases.md`.
