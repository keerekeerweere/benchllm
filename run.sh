#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STACK_ROOT="$ROOT_DIR/runtime/dual-3090"
PYTHON_TOOL="uv"
CATALOG_PATH="$ROOT_DIR/catalogs/dual-3090-openai.yaml"
DRY_RUN=0
BENCHLLM_REPO="${BENCHLLM_REPO:-$ROOT_DIR}"
VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
LLAMACPP_REPO="${LLAMACPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"

usage() {
  cat <<'EOF'
Usage: bash run.sh [--root PATH] [--python-tool uv|venv] [--catalog PATH] [--dry-run]

Bootstraps a self-contained runtime folder for benchllm, vLLM, and llama.cpp.
EOF
}

log() {
  printf '%s\n' "$*"
}

resolve_python_tool() {
  if [[ "$PYTHON_TOOL" == "uv" ]] && ! command -v uv >/dev/null 2>&1; then
    log "= uv not found, falling back to venv"
    PYTHON_TOOL="venv"
  fi
}

load_env_file() {
  local path="$1"
  if [[ -f "$path" ]]; then
    log "= load env $path"
    set -a
    # shellcheck disable=SC1090
    source "$path"
    set +a
  fi
}

run_cmd() {
  log "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

run_shell() {
  log "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    bash -lc "$*"
  fi
}

clone_if_missing() {
  local repo="$1"
  local dest="$2"
  if [[ -d "$dest/.git" || -d "$dest" ]]; then
    log "= skip existing $dest"
    return
  fi
  run_cmd git clone "$repo" "$dest"
}

create_venv_if_missing() {
  local tool="$1"
  local dest="$2"
  if [[ -x "$dest/bin/python" ]]; then
    log "= skip existing venv $dest"
    return
  fi
  if [[ "$tool" == "uv" ]]; then
    run_cmd uv venv "$dest"
  else
    run_cmd python3 -m venv "$dest"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      STACK_ROOT="$2"
      shift 2
      ;;
    --python-tool)
      PYTHON_TOOL="$2"
      shift 2
      ;;
    --catalog)
      CATALOG_PATH="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$PYTHON_TOOL" != "uv" && "$PYTHON_TOOL" != "venv" ]]; then
  printf 'Unsupported --python-tool: %s\n' "$PYTHON_TOOL" >&2
  exit 1
fi

resolve_python_tool
load_env_file "$ROOT_DIR/.env"
load_env_file "$STACK_ROOT/.env"
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"
fi

mkdir_cmd=(mkdir -p "$STACK_ROOT/src" "$STACK_ROOT/.venvs" "$STACK_ROOT/cache" "$STACK_ROOT/logs" "$STACK_ROOT/results")
run_cmd "${mkdir_cmd[@]}"
if [[ ! -f "$STACK_ROOT/.env.example" ]]; then
  run_cmd cp "$ROOT_DIR/.env.example" "$STACK_ROOT/.env.example"
fi
if [[ ! -f "$STACK_ROOT/.env" ]]; then
  run_cmd cp "$ROOT_DIR/.env.example" "$STACK_ROOT/.env"
fi

clone_if_missing "$BENCHLLM_REPO" "$STACK_ROOT/src/benchllm"
clone_if_missing "$VLLM_REPO" "$STACK_ROOT/src/vllm"
clone_if_missing "$LLAMACPP_REPO" "$STACK_ROOT/src/llama.cpp"

if [[ "$PYTHON_TOOL" == "uv" ]]; then
  create_venv_if_missing uv "$STACK_ROOT/.venvs/benchllm"
  create_venv_if_missing uv "$STACK_ROOT/.venvs/vllm"
  run_cmd uv pip install --python "$STACK_ROOT/.venvs/benchllm/bin/python" -e "$STACK_ROOT/src/benchllm"
  run_cmd uv pip install --python "$STACK_ROOT/.venvs/vllm/bin/python" -e "$STACK_ROOT/src/vllm"
  PREPARE_PYTHON="$STACK_ROOT/.venvs/benchllm/bin/python"
else
  create_venv_if_missing venv "$STACK_ROOT/.venvs/benchllm"
  create_venv_if_missing venv "$STACK_ROOT/.venvs/vllm"
  run_cmd "$STACK_ROOT/.venvs/benchllm/bin/python" -m pip install --upgrade pip
  run_cmd "$STACK_ROOT/.venvs/vllm/bin/python" -m pip install --upgrade pip
  run_cmd "$STACK_ROOT/.venvs/benchllm/bin/python" -m pip install -e "$STACK_ROOT/src/benchllm"
  run_cmd "$STACK_ROOT/.venvs/vllm/bin/python" -m pip install -e "$STACK_ROOT/src/vllm"
  PREPARE_PYTHON="$STACK_ROOT/.venvs/benchllm/bin/python"
fi

run_cmd cmake -S "$STACK_ROOT/src/llama.cpp" -B "$STACK_ROOT/src/llama.cpp/build" -DGGML_CUDA=ON
run_cmd cmake --build "$STACK_ROOT/src/llama.cpp/build" -j

run_cmd "$PREPARE_PYTHON" -m benchllm prepare --catalog "$CATALOG_PATH" --output-dir "$STACK_ROOT"

log ""
log "Prepared stack root: $STACK_ROOT"
log "Runtime launchers:"
log "  $STACK_ROOT/launchers/vllm-qwen3-coder-next-fp8.sh"
log "  $STACK_ROOT/launchers/vllm-devstral-small.sh"
log "  $STACK_ROOT/launchers/vllm-qwen-moe-awq.sh"
log "  $STACK_ROOT/launchers/llamacpp-devstral-q4km.sh"
