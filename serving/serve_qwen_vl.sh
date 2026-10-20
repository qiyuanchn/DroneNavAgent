#!/usr/bin/env bash
# Serve a Qwen-VL checkpoint with vLLM for the navagent OpenAI backend.
#
# The script is host-agnostic: it runs vLLM either in the current environment or
# inside a container (`--container` / `CONTAINER=...`), and every path is a flag
# or an environment variable. It deliberately contains no host-specific
# container name, workdir or checkpoint path.
#
# Examples:
#   ./serving/serve_qwen_vl.sh --model 8b --gpu 0 --port 8000
#   ./serving/serve_qwen_vl.sh --model 27b --gpu 1 --port 8010 \
#       --served-model-name qwen3.8-27b-vl
#   CONTAINER=vlm-nav ./serving/serve_qwen_vl.sh --model 8b --port 8000 \
#       --vllm-python /opt/vllm/bin/python
#
# Checkpoint aliases map to environment overrides:
#   QWEN3_VL_8B_PATH     default /models_ro/Qwen3-VL-8B
#   QWEN38_27B_VL_PATH   default /qwen27b
#   QWEN_VL_MODEL        default checkpoint when --model is omitted
set -euo pipefail

DEFAULT_8B="${QWEN3_VL_8B_PATH:-/models_ro/Qwen3-VL-8B}"
DEFAULT_27B="${QWEN38_27B_VL_PATH:-/qwen27b}"

MODEL="${QWEN_VL_MODEL:-$DEFAULT_8B}"
SERVED_NAME=""
GPU="${CUDA_VISIBLE_DEVICES:-0}"
HOST="${QWEN_VL_HOST:-0.0.0.0}"
PORT="${QWEN_VL_PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
LIMIT_MM_IMAGE="${LIMIT_MM_IMAGE:-4}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
PREFIX_CACHING="${PREFIX_CACHING:-1}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-1}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-0}"
MM_PROCESSOR_CACHE_GB="${MM_PROCESSOR_CACHE_GB:-}"
HF_HOME="${HF_HOME:-}"
VLLM_PYTHON="${VLLM_PYTHON:-python}"
WORKDIR=""
LOG=""
CONTAINER="${CONTAINER:-}"
DETACH=1
LANGUAGE_MODEL_ONLY=0
THINKING=0
REASONING_PARSER="qwen3"
TOOL_CALL_PARSER="qwen3_coder"
API_KEY="${VLLM_API_KEY:-}"
API_KEY_FILE=""
EXTRA_ARGS=()

usage() {
    sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  -m, --model PATH|8b|27b     checkpoint directory or alias
  -n, --served-model-name N   model name clients send (default from alias/basename)
  -g, --gpu IDS               CUDA_VISIBLE_DEVICES value (default 0)
  -p, --port PORT             HTTP port (default 8000)
      --host ADDR             bind address (default 0.0.0.0)
      --gpu-mem-util FLOAT    vLLM GPU memory fraction (default 0.90)
      --max-model-len N       context length (default 16384)
      --max-num-seqs N        concurrent sequences (default 8)
      --tensor-parallel N     tensor parallel size (default 1)
      --limit-mm-image N      images per prompt (default 4, vision only)
      --prefix-caching on|off enable vLLM prefix caching (default on)
      --chunked-prefill on|off enable chunked prefill (default on)
      --async-scheduling      opt in to vLLM async scheduling (newer builds)
      --mm-processor-cache-gb N
                              cache preprocessed images (vLLM build permitting)
      --no-prefix-caching     alias for --prefix-caching off
      --no-chunked-prefill    alias for --chunked-prefill off
      --hf-home PATH          Hugging Face cache
      --vllm-python PATH      interpreter that has vLLM (default python)
      --workdir PATH          cd before launch
      --log PATH              server log (default /tmp/qwen-vl-<port>.log; '-' = stdout)
      --container NAME        run via docker exec in NAME (or CONTAINER env)
      --api-key KEY           set VLLM_API_KEY
      --api-key-file PATH     read VLLM_API_KEY from PATH
      --language-model-only   drop the vision tower (text-only serving)
      --thinking              enable Qwen thinking (default off)
      --reasoning-parser P    default qwen3
      --tool-call-parser P    default qwen3_coder
      --foreground            do not detach (default: detach)
      --dry-run               print the command and exit
      --extra ARG             extra vLLM argument, repeatable
  -h, --help                  show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model) MODEL="$2"; shift 2 ;;
        -n|--served-model-name) SERVED_NAME="$2"; shift 2 ;;
        -g|--gpu) GPU="$2"; shift 2 ;;
        -p|--port) PORT="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --gpu-mem-util) GPU_MEM_UTIL="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
        --tensor-parallel) TENSOR_PARALLEL="$2"; shift 2 ;;
        --limit-mm-image) LIMIT_MM_IMAGE="$2"; shift 2 ;;
        --prefix-caching) PREFIX_CACHING="$2"; shift 2 ;;
        --no-prefix-caching) PREFIX_CACHING=0; shift ;;
        --chunked-prefill) CHUNKED_PREFILL="$2"; shift 2 ;;
        --no-chunked-prefill) CHUNKED_PREFILL=0; shift ;;
        --async-scheduling) ASYNC_SCHEDULING=1; shift ;;
        --mm-processor-cache-gb) MM_PROCESSOR_CACHE_GB="$2"; shift 2 ;;
        --hf-home) HF_HOME="$2"; shift 2 ;;
        --vllm-python) VLLM_PYTHON="$2"; shift 2 ;;
        --workdir) WORKDIR="$2"; shift 2 ;;
        --log) LOG="$2"; shift 2 ;;
        --container) CONTAINER="$2"; shift 2 ;;
        --api-key) API_KEY="$2"; shift 2 ;;
        --api-key-file) API_KEY_FILE="$2"; shift 2 ;;
        --language-model-only) LANGUAGE_MODEL_ONLY=1; shift ;;
        --thinking) THINKING=1; shift ;;
        --reasoning-parser) REASONING_PARSER="$2"; shift 2 ;;
        --tool-call-parser) TOOL_CALL_PARSER="$2"; shift 2 ;;
        --foreground) DETACH=0; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --extra) EXTRA_ARGS+=("$2"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
DRY_RUN="${DRY_RUN:-0}"

case "$MODEL" in
    8b|qwen3-vl-8b) MODEL="$DEFAULT_8B" ;;
    27b|qwen3.8-27b-vl|qwen3-27b-vl) MODEL="$DEFAULT_27B" ;;
esac

if [[ -z "$SERVED_NAME" ]]; then
    case "$MODEL" in
        "$DEFAULT_8B") SERVED_NAME="qwen3-vl-8b" ;;
        "$DEFAULT_27B") SERVED_NAME="qwen3.8-27b-vl" ;;
        *) SERVED_NAME="$(basename "${MODEL%/}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._-' '-')" ;;
    esac
fi

if [[ -z "$LOG" ]]; then
    LOG="/tmp/qwen-vl-${PORT}.log"
fi

if [[ -n "$API_KEY_FILE" ]]; then
    API_KEY="$(cat "$API_KEY_FILE")"
fi

CMD=(
    "$VLLM_PYTHON" -m vllm serve "$MODEL"
    --host "$HOST"
    --port "$PORT"
    --served-model-name "$SERVED_NAME"
    --tensor-parallel-size "$TENSOR_PARALLEL"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --reasoning-parser "$REASONING_PARSER"
    --enable-auto-tool-choice
    --tool-call-parser "$TOOL_CALL_PARSER"
)
if [[ "$PREFIX_CACHING" == "1" ]]; then
    CMD+=(--enable-prefix-caching)
fi
if [[ "$CHUNKED_PREFILL" == "1" ]]; then
    CMD+=(--enable-chunked-prefill)
fi
if [[ "$ASYNC_SCHEDULING" == "1" ]]; then
    CMD+=(--async-scheduling)
fi
if [[ -n "$MM_PROCESSOR_CACHE_GB" && "$MM_PROCESSOR_CACHE_GB" != "0" ]]; then
    CMD+=(--mm-processor-cache-gb "$MM_PROCESSOR_CACHE_GB")
fi
if [[ "$LANGUAGE_MODEL_ONLY" == "1" ]]; then
    CMD+=(--language-model-only)
else
    CMD+=(--limit-mm-per-prompt "{\"image\": ${LIMIT_MM_IMAGE}}")
fi
if [[ "$THINKING" == "1" ]]; then
    CMD+=(--default-chat-template-kwargs '{"enable_thinking": true}')
else
    CMD+=(--default-chat-template-kwargs '{"enable_thinking": false}')
fi
CMD+=("${EXTRA_ARGS[@]}")

quote() { printf '%q' "$1"; }
CMD_LINE=""
for arg in "${CMD[@]}"; do
    CMD_LINE+=" $(quote "$arg")"
done

ENV_LINE="CUDA_VISIBLE_DEVICES=$(quote "$GPU")"
if [[ -n "$HF_HOME" ]]; then
    ENV_LINE+=" HF_HOME=$(quote "$HF_HOME")"
fi
if [[ -n "$API_KEY" ]]; then
    ENV_LINE+=" VLLM_API_KEY=$(quote "$API_KEY")"
fi

INNER=""
if [[ -n "$WORKDIR" ]]; then
    INNER+="cd $(quote "$WORKDIR") && "
fi
INNER+="export ${ENV_LINE# } && exec ${CMD_LINE# }"
if [[ "$LOG" != "-" ]]; then
    INNER+=" >> $(quote "$LOG") 2>&1"
fi

if [[ "$DRY_RUN" == "1" ]]; then
    if [[ -n "$CONTAINER" ]]; then
        printf 'docker exec %s%s bash -lc %q\n' \
            "$([[ "$DETACH" == "1" ]] && echo '-d ')" "$CONTAINER" "$INNER"
    else
        printf 'bash -lc %q\n' "$INNER"
    fi
    exit 0
fi

if [[ -n "$CONTAINER" ]]; then
    if [[ "$DETACH" == "1" ]]; then
        docker exec -d "$CONTAINER" bash -lc "$INNER"
    else
        docker exec "$CONTAINER" bash -lc "$INNER"
    fi
else
    if [[ "$DETACH" == "1" ]]; then
        nohup bash -lc "$INNER" >/dev/null 2>&1 &
        echo "started pid $!"
    else
        bash -lc "$INNER"
    fi
fi

echo "model:        $MODEL"
echo "served name:  $SERVED_NAME"
echo "endpoint:     http://127.0.0.1:${PORT}/v1"
echo "gpu:          $GPU"
echo "vision:       $([[ "$LANGUAGE_MODEL_ONLY" == "1" ]] && echo off || echo on)"
echo "thinking:     $([[ "$THINKING" == "1" ]] && echo on || echo off)"
echo "prefix cache: $([[ "$PREFIX_CACHING" == "1" ]] && echo on || echo off)"
echo "chunked pref: $([[ "$CHUNKED_PREFILL" == "1" ]] && echo on || echo off)"
echo "async sched:  $([[ "$ASYNC_SCHEDULING" == "1" ]] && echo on || echo off)"
echo "mm cache:     ${MM_PROCESSOR_CACHE_GB:-off} GiB"
echo "log:          $LOG"
echo
echo "navagent:"

cat <<EOF
  python scripts/run_vlm_nav_tools.py --backend openai \\
      --model ${SERVED_NAME} --base-url http://127.0.0.1:${PORT}/v1 ...
EOF
if [[ -z "$API_KEY" ]]; then
    echo "  (server has no API key; set OPENAI_API_KEY to any non-empty value if required)"
else
    echo "  OPENAI_API_KEY must match --api-key / VLLM_API_KEY"
fi
