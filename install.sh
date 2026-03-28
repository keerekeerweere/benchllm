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
