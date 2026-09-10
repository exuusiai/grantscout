#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_PATH="${PAPERSCOUT_MODEL_PATH:-/root/autodl-tmp/models/Qwen-Qwen3-8B}"
MODEL_NAME="${PAPERSCOUT_MODEL_NAME:-Qwen/Qwen3-8B}"
HOST="${PAPERSCOUT_MODEL_HOST:-127.0.0.1}"
PORT="${PAPERSCOUT_MODEL_PORT:-8001}"
GPU_UTILIZATION="${PAPERSCOUT_GPU_MEMORY_UTILIZATION:-0.88}"
MAX_MODEL_LEN="${PAPERSCOUT_MODEL_MAX_LENGTH:-8192}"
PYTHON_BIN="${PAPERSCOUT_PYTHON:-$PROJECT_DIR/.venv-vllm/bin/python}"
USE_FLASHINFER_SAMPLER="${PAPERSCOUT_USE_FLASHINFER_SAMPLER:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Configured vLLM Python is not executable: $PYTHON_BIN" >&2
  echo "Set PAPERSCOUT_PYTHON to the environment where vLLM is installed." >&2
  exit 1
fi

case "$HOST" in
  localhost|127.0.0.1|::1) ;;
  *)
    echo "PAPERSCOUT_MODEL_HOST must be a loopback address, not: $HOST" >&2
    exit 1
    ;;
esac

case "$USE_FLASHINFER_SAMPLER" in
  0|1) ;;
  *)
    echo "PAPERSCOUT_USE_FLASHINFER_SAMPLER must be 0 or 1." >&2
    exit 1
    ;;
esac

# FlashInfer's sampler in the current RTX 5090 environment rejects sm_120
# during JIT setup. Keep the compatible vLLM sampler as the safe default.
export VLLM_USE_FLASHINFER_SAMPLER="$USE_FLASHINFER_SAMPLER"

"$PYTHON_BIN" "$SCRIPT_DIR/preflight_vllm.py" --model-path "$MODEL_PATH"

exec "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN"
