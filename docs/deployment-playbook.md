# Dual RTX 3090 Deployment Playbook

This playbook turns the benchmark plan into concrete launch profiles for a workstation with two NVLinked RTX 3090 24 GB cards on PCIe 4.0 x16.

## Primary recommendations

- Use `vLLM` as the primary OpenAI-compatible backend for current FP8, AWQ, and MoE checkpoints.
- Use `llama.cpp` as the GGUF fallback path and for targeted constrained-decoding comparisons.
- Keep embeddings separate from generation. The default profile is a dedicated embedding process using `Qwen/Qwen3-Embedding-4B`.
- Optimize for `8k` context and `2-4` concurrent interactive sessions first. Expand context only after measuring TTFT and structured-output regressions.

## Recommended vLLM knobs

- `--tensor-parallel-size 2` for both 3090s.
- `--gpu-memory-utilization 0.92` as the first-pass ceiling.
- `--enable-prefix-caching` for shared-prefix agent prompts.
- `--enable-chunked-prefill` for mixed prompt lengths.
- Keep speculative decoding opt-in and benchmark-gated rather than default.
- Treat disaggregated prefill or exotic kernels as follow-up experiments, not the baseline profile on one workstation.

## Recommended llama.cpp knobs

- `--flash-attn` should be baseline on CUDA builds.
- Compare `--split-mode row` against `layer` only if row split under-utilizes one GPU.
- Start with `--cache-type-k q8_0 --cache-type-v q8_0` for an 8k interactive profile.
- Use `--parallel 4` only if JSON reliability and tail latency stay acceptable.

## Practical run order

1. Start the embedding service and confirm it is isolated from main inference runs.
2. Benchmark `vllm-qwen3-coder-next-fp8` and `vllm-devstral-small` first.
3. Benchmark the AWQ MoE profile only after the first two establish a good latency baseline.
4. Benchmark the `llama.cpp` GGUF fallback once the vLLM winners are known.
5. Re-run the top 2 profiles with cold starts and then with warm caches.

## Interpretation guidance

- Prefer the profile with the best combined TTFT, JSON pass rate, and usable 2-session behavior over the absolute top decode throughput number.
- Reject any profile that wins on throughput but produces unstable JSON or large p95 latency spikes.
- Treat `MiniMax`-class checkpoints as stretch experiments only if they materially fit the local memory envelope and have a clean OpenAI-compatible path.
