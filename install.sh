#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
  DEFAULT_INSTALL_ROOT="/opt/benchllm"
else
  DEFAULT_INSTALL_ROOT="${HOME:-/tmp}/.local/share/benchllm"
fi

INSTALL_ROOT="${BENCHLLM_INSTALL_ROOT:-$DEFAULT_INSTALL_ROOT}"
REPO_URL="${BENCHLLM_REPO_URL:-https://github.com/keerekeerweere/benchllm}"
REPO_BRANCH="${BENCHLLM_REPO_BRANCH:-main}"
PYTHON_TOOL="${BENCHLLM_PYTHON_TOOL:-uv}"
DRY_RUN="${INSTALL_DRY_RUN:-0}"
REPO_DIR="$INSTALL_ROOT/src/benchllm"
RUNTIME_ROOT="$INSTALL_ROOT/runtime"

log() {
  printf '%s\n' "$*"
}

run_cmd() {
  log "+ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

add_local_bin_to_path() {
  local home_dir="${HOME:-/root}"
  case ":$PATH:" in
    *":$home_dir/.local/bin:"*) ;;
    *) export PATH="$home_dir/.local/bin:$PATH" ;;
  esac
}

ensure_system_deps() {
  local missing=0
  for cmd in git cmake python3; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
      missing=1
    fi
  done
  if [[ "$missing" -eq 0 ]]; then
    return
  fi
  if [[ "${EUID:-$(id -u)}" -ne 0 ]] || ! command -v apt-get >/dev/null 2>&1; then
    log "! missing required system packages (git/cmake/python3) and cannot install automatically"
    exit 1
  fi
  run_cmd apt-get update
  run_cmd apt-get install -y ca-certificates curl git cmake build-essential python3 python3-venv
}

install_uv_with_pipx_if_possible() {
  add_local_bin_to_path
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi
  if command -v pipx >/dev/null 2>&1; then
    run_cmd pipx install uv
    add_local_bin_to_path
    if [[ "$DRY_RUN" -eq 1 ]] || command -v uv >/dev/null 2>&1; then
      return 0
    fi
  fi
  if [[ "${EUID:-$(id -u)}" -eq 0 ]] && command -v apt-get >/dev/null 2>&1; then
    run_cmd apt-get update
    run_cmd apt-get install -y pipx python3-venv
    add_local_bin_to_path
    run_cmd pipx install uv
    add_local_bin_to_path
    if [[ "$DRY_RUN" -eq 1 ]] || command -v uv >/dev/null 2>&1; then
      return 0
    fi
  fi
  return 1
}

resolve_python_tool() {
  if [[ "$PYTHON_TOOL" == "uv" ]] && ! command -v uv >/dev/null 2>&1; then
    if install_uv_with_pipx_if_possible; then
      log "= uv will be managed via pipx"
      return
    fi
    log "= uv not found, falling back to venv"
    PYTHON_TOOL="venv"
  fi
}

clone_or_update_repo() {
  if [[ -d "$REPO_DIR/.git" ]]; then
    log "= reuse existing repo $REPO_DIR"
    run_cmd git -C "$REPO_DIR" fetch --all --tags
    run_cmd git -C "$REPO_DIR" checkout "$REPO_BRANCH"
    run_cmd git -C "$REPO_DIR" pull --ff-only
    return
  fi
  run_cmd mkdir -p "$INSTALL_ROOT/src"
  run_cmd git clone "$REPO_URL" "$REPO_DIR"
  run_cmd git -C "$REPO_DIR" checkout "$REPO_BRANCH"
}

sync_env_files() {
  if [[ ! -f "$INSTALL_ROOT/.env.example" && -f "$REPO_DIR/.env.example" ]]; then
    run_cmd cp "$REPO_DIR/.env.example" "$INSTALL_ROOT/.env.example"
  fi
  if [[ ! -f "$INSTALL_ROOT/.env" && -f "$REPO_DIR/.env.example" ]]; then
    run_cmd cp "$REPO_DIR/.env.example" "$INSTALL_ROOT/.env"
  fi
  if [[ -f "$INSTALL_ROOT/.env" ]]; then
    run_cmd cp "$INSTALL_ROOT/.env" "$REPO_DIR/.env"
  fi
}

ensure_system_deps
resolve_python_tool
clone_or_update_repo
sync_env_files

bootstrap_cmd=(bash "$REPO_DIR/run.sh" --root "$RUNTIME_ROOT" --python-tool "$PYTHON_TOOL")
if [[ "$DRY_RUN" -eq 1 ]]; then
  bootstrap_cmd+=(--dry-run)
fi
run_cmd "${bootstrap_cmd[@]}"

log ""
log "Installer complete."
log "Install root: $INSTALL_ROOT"
log "Edit $INSTALL_ROOT/.env to add HF_TOKEN and other machine-specific overrides."
