from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Iterable

from benchllm.catalog import Profile, load_catalog


ENV_EXAMPLE = """# Copy to .env and fill in values that apply to your machine.
# Hugging Face token for gated/private models and higher-rate downloads.
HF_TOKEN=

# Override cache/model locations if needed.
HF_HOME=
MODEL_ROOT=

# GPU/runtime defaults for the dual RTX 3090 target.
CUDA_VISIBLE_DEVICES=0,1
VLLM_WORKER_MULTIPROC_METHOD=spawn

# Optional command overrides for generated launchers.
VLLM_PYTHON_BIN=
LLAMA_CPP_SERVER_BIN=
LLAMA_CPP_ROOT=
"""


def prepare_runtime_bundle(
    catalog_path: str | Path,
    output_dir: str | Path,
) -> Path:
    catalog = load_catalog(catalog_path)
    manifest = {
        "catalog_path": str(Path(catalog_path).resolve()),
        "profiles": list(catalog.profiles.keys()),
        "workloads": list(catalog.workloads.keys()),
        "matrix": {
            "profiles": catalog.matrix.profiles,
            "workloads": catalog.matrix.workloads,
            "concurrencies": catalog.matrix.concurrencies,
            "repetitions": catalog.matrix.repetitions,
        },
    }
    return prepare_runtime_bundle_from_profiles(
        catalog.profiles.values(),
        output_dir,
        manifest=manifest,
    )


def prepare_runtime_bundle_from_profiles(
    profiles: Iterable[Profile],
    output_dir: str | Path,
    *,
    manifest: dict[str, object] | None = None,
) -> Path:
    root = Path(output_dir)
    launchers_dir = root / "launchers"
    manifests_dir = root / "manifests"
    logs_dir = root / "logs"
    results_dir = root / "results"

    for directory in (root, launchers_dir, manifests_dir, logs_dir, results_dir):
        directory.mkdir(parents=True, exist_ok=True)

    _write_env_script(root)
    _write_env_example(root)
    profile_ids: list[str] = []
    for profile in profiles:
        profile_ids.append(profile.id)
        if profile.launch is None:
            continue
        launcher_path = launchers_dir / f"{profile.id}.sh"
        launcher_path.write_text(_render_launcher(profile), encoding="utf-8")
        launcher_path.chmod(0o755)
    runtime_manifest = manifest or {"profiles": profile_ids}
    (manifests_dir / "runtime-manifest.json").write_text(json.dumps(runtime_manifest, indent=2), encoding="utf-8")
    return root


def _write_env_script(root: Path) -> None:
    env_script = root / "env.sh"
    env_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
                'if [ -f "$ROOT_DIR/.env" ]; then',
                "  set -a",
                '  source "$ROOT_DIR/.env"',
                "  set +a",
                "fi",
                'export BENCHLLM_STACK_ROOT="$ROOT_DIR"',
                'export HF_HOME="${HF_HOME:-$ROOT_DIR/cache/huggingface}"',
                'if [ -n "${HF_TOKEN:-}" ]; then',
                '  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"',
                '  export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"',
                "fi",
                'export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"',
                'export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"',
                'mkdir -p "$ROOT_DIR/cache/huggingface" "$ROOT_DIR/logs" "$ROOT_DIR/results"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    env_script.chmod(0o755)


def _write_env_example(root: Path) -> None:
    env_example = root / ".env.example"
    env_example.write_text(ENV_EXAMPLE, encoding="utf-8")


def _render_launcher(profile: Profile) -> str:
    assert profile.launch is not None
    command = _resolve_command(profile)
    args = " ".join(shlex.quote(value) for value in profile.launch.args)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"',
        'source "$ROOT_DIR/env.sh"',
    ]
    if profile.backend == "vllm":
        lines.append('if [ -f "$ROOT_DIR/.venvs/vllm/bin/activate" ]; then source "$ROOT_DIR/.venvs/vllm/bin/activate"; fi')
    elif profile.backend == "llama.cpp":
        lines.append('export LLAMA_CPP_ROOT="${LLAMA_CPP_ROOT:-$ROOT_DIR/src/llama.cpp}"')
    for key, value in profile.launch.env.items():
        lines.append(f"export {key}={shlex.quote(value)}")
    lines.append(f'exec {command} {args}'.rstrip())
    lines.append("")
    return "\n".join(lines)


def _resolve_command(profile: Profile) -> str:
    assert profile.launch is not None
    command = profile.launch.command
    if profile.backend == "vllm" and command[:3] == ["python", "-m", "vllm.entrypoints.openai.api_server"]:
        return '"${VLLM_PYTHON_BIN:-$ROOT_DIR/.venvs/vllm/bin/python}" -m vllm.entrypoints.openai.api_server'
    if profile.backend == "llama.cpp" and command == ["./llama-server"]:
        return '"${LLAMA_CPP_SERVER_BIN:-$ROOT_DIR/src/llama.cpp/build/bin/llama-server}"'
    return " ".join(shlex.quote(value) for value in command)
