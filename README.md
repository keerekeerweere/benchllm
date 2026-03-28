# benchllm

`benchllm` is a small benchmark and deployment-playbook project for comparing local LLM serving profiles on a dual RTX 3090 workstation.

## What it does

- Loads a YAML catalog of inference profiles, workloads, and run-matrix settings.
- Plans benchmark runs for `vLLM` and `llama.cpp` style OpenAI-compatible endpoints.
- Executes streaming `/chat/completions` requests and records TTFT, total latency, tokens, decode throughput, and structured-output success.
- Summarizes results by profile and workload.

## Quick start

```bash
python3 -m unittest discover -s tests -v
python3 -m benchllm.cli plan --catalog catalogs/dual-3090-openai.yaml
python3 -m benchllm.cli prepare --catalog catalogs/dual-3090-openai.yaml --output-dir runtime/dual-3090
python3 -m benchllm.cli run --catalog catalogs/dual-3090-openai.yaml --output results.json
python3 -m benchllm.cli summarize --results results.json
```

## Bootstrap the full stack

```bash
cp .env.example .env
# edit .env and set HF_TOKEN if you need gated/private Hugging Face access
bash run.sh --root runtime/dual-3090 --python-tool uv
```

This bootstrap flow creates a self-contained runtime folder, clones `vllm` and `llama.cpp`, installs `benchllm`, prepares runtime launchers, and leaves you with generated scripts under `runtime/dual-3090/launchers/`.

For a no-argument installer suitable for `curl | bash`, use:

```bash
curl -fsSL https://raw.githubusercontent.com/keerekeerweere/benchllm/main/install.sh | sudo bash
```

It defaults to `/opt/benchllm` under `sudo`, clones this repo, creates `/opt/benchllm/.env`, and then runs the normal bootstrap flow.

## Notes

- The tool assumes an OpenAI-compatible local endpoint for inference.
- `.env` is used for secrets and machine-specific overrides such as `HF_TOKEN`, `HF_HOME`, `CUDA_VISIBLE_DEVICES`, and launcher overrides.
- Embeddings are intended to be served separately from generation and are documented in the sample catalog and playbook.
- The included catalog is opinionated for a dual 3090 NVLink setup and should be edited to match the exact model paths and backend builds on the target machine.
