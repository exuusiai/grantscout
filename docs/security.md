# Security And Reliability

- Paper text is treated as untrusted data. It is never executed as Python, shell,
  or a tool instruction.
- Tool inputs are bounded by Pydantic settings and CLI limits.
- The local model client accepts only loopback endpoints (`localhost`, `127.0.0.1`,
  or `::1`) and sends only configured prompts there. It rejects credentials,
  query strings, fragments, third-party OpenAI-compatible gateways, and any
  `/responses` path before making a request. HTTP redirects are disabled.
- `scripts/preflight_vllm.py` validates model shards and rejects non-loopback
  post-start endpoint probes before any request is made.
- `scripts/start_vllm.sh` rejects a non-loopback model bind address, keeping the
  local inference server private by default.
- Reports preserve warnings when evidence is missing or outcome language conflicts.
- Dataset manifests record source URLs, download timestamps, and SHA256 hashes.
- Do not put API keys in the repository; use `.env`, which is ignored by Git.
