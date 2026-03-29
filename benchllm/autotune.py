from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time
from typing import Any

import httpx

from benchllm.catalog import BenchmarkRunSpec, LaunchSpec, Profile, ValidationRules, Workload, load_catalog
from benchllm.reporting import summarize_results
from benchllm.runner import BenchmarkResult, BenchmarkRunner


SUPPORTED_MACHINES = {"dual-3090": "dual-3090"}
SUPPORTED_STRATEGIES = {"fast-agentic": "fast-agentic"}

_OPTIMIZATION_REGISTRY: dict[str, dict[str, Any]] = {
    "prefix-caching": {"min_arch": "ampere", "maturity": "stable", "memory_delta_gb": -0.3},
    "chunked-prefill": {"min_arch": "ampere", "maturity": "stable", "memory_delta_gb": -0.2},
    "q8-kv-cache": {"min_arch": "ampere", "maturity": "stable", "memory_delta_gb": -0.8},
    "ream": {"min_arch": "ampere", "maturity": "experimental", "memory_delta_gb": -1.2},
    "turboquant-3": {"min_arch": "ampere", "maturity": "experimental", "memory_delta_gb": -1.5},
    "flash-attn": {"min_arch": "ampere", "maturity": "stable", "memory_delta_gb": -0.2},
}


@dataclass(frozen=True)
class MachineProfile:
    id: str
    gpu_name: str
    gpu_count: int
    vram_per_gpu_gb: float
    nvlink: bool
    architecture: str
    tensor_parallel_size: int
    supported_backends: tuple[str, ...]
    preferred_context_tiers: tuple[int, ...]
    preferred_concurrency_tiers: tuple[int, ...]
    gpu_memory_utilization: float = 0.92

    @property
    def effective_vram_per_gpu_gb(self) -> float:
        return round(self.vram_per_gpu_gb * self.gpu_memory_utilization, 3)


@dataclass(frozen=True)
class SearchStrategy:
    id: str
    use_case: str
    ranking_mode: str
    max_candidates: int
    context_tiers: tuple[int, ...]
    concurrency_tiers: tuple[int, ...]
    backend_order: tuple[str, ...]
    quant_order: tuple[str, ...]
    experimental_optimizations: tuple[str, ...]
    stop_after_viable: int = 2


@dataclass(frozen=True)
class CandidateVariant:
    id: str
    family: str
    model: str
    backend: str
    quantization: str
    context_size: int
    concurrency: int
    source: str
    size_billions: float | None = None
    remote: bool = True
    optimizations: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProbeResult:
    status: str
    estimated_vram_per_gpu_gb: float
    estimated_ttft_ms: float
    estimated_decode_tokens_per_second: float
    structured_output_reliable: bool
    rejection_reason: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: CandidateVariant
    profile: Profile
    probe: ProbeResult
    score: float
    rank_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "profile": {
                "id": self.profile.id,
                "backend": self.profile.backend,
                "kind": self.profile.kind,
                "model": self.profile.model,
                "api_base": self.profile.api_base,
                "launch": asdict(self.profile.launch) if self.profile.launch else None,
                "metadata": self.profile.metadata,
            },
            "probe": self.probe.to_dict(),
            "score": self.score,
            "rank_reason": self.rank_reason,
        }


@dataclass(frozen=True)
class DeploymentRecommendation:
    machine: MachineProfile
    strategy: SearchStrategy
    winner: CandidateEvaluation
    runner_up: CandidateEvaluation | None
    rejected: tuple[CandidateEvaluation, ...]
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "machine": asdict(self.machine),
            "strategy": asdict(self.strategy),
            "winner": self.winner.to_dict(),
            "runner_up": self.runner_up.to_dict() if self.runner_up else None,
            "rejected": [item.to_dict() for item in self.rejected],
            "generated_at": self.generated_at,
        }


@dataclass(frozen=True)
class LiveBenchmarkConfig:
    enabled: bool = True
    startup_timeout_seconds: float = 150.0
    warmup_timeout_seconds: float = 45.0
    log_dir: str | None = None


def get_machine_profile(machine_id: str) -> MachineProfile:
    if machine_id not in SUPPORTED_MACHINES:
        raise ValueError(f"Unsupported machine profile: {machine_id}")
    return MachineProfile(
        id="dual-3090",
        gpu_name="RTX 3090",
        gpu_count=2,
        vram_per_gpu_gb=24.0,
        nvlink=True,
        architecture="ampere",
        tensor_parallel_size=2,
        supported_backends=("vllm", "llama.cpp"),
        preferred_context_tiers=(8192, 16384, 32768),
        preferred_concurrency_tiers=(1, 2, 4),
    )


def get_search_strategy(strategy_id: str) -> SearchStrategy:
    if strategy_id not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unsupported strategy: {strategy_id}")
    return SearchStrategy(
        id="fast-agentic",
        use_case="coding",
        ranking_mode="responsiveness-first",
        max_candidates=8,
        context_tiers=(8192, 16384),
        concurrency_tiers=(1, 2, 4),
        backend_order=("vllm", "llama.cpp"),
        quant_order=("fp8", "awq-4bit", "gptq-4bit", "q4_k_m", "q6_k", "q8_0"),
        experimental_optimizations=("ream", "turboquant-3"),
    )


class LLMFitRecommender:
    def __init__(self, command: list[str] | None = None) -> None:
        self._command = command or ["llmfit", "recommend", "--json", "--use-case", "coding"]

    def recommend(self, machine: MachineProfile, strategy: SearchStrategy) -> list[CandidateVariant]:
        try:
            proc = subprocess.run(
                self._command,
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return _curated_candidates(machine, strategy)

        parsed = _parse_llmfit_payload(proc.stdout, strategy)
        if not parsed:
            return _curated_candidates(machine, strategy)
        return parsed[: strategy.max_candidates]


class HeuristicProbe:
    def probe(self, candidate: CandidateVariant, machine: MachineProfile) -> ProbeResult:
        unsupported = [
            optimization
            for optimization in candidate.optimizations
            if not _is_optimization_supported(optimization, machine.architecture)
        ]
        if unsupported:
            return ProbeResult(
                status="rejected",
                estimated_vram_per_gpu_gb=0.0,
                estimated_ttft_ms=0.0,
                estimated_decode_tokens_per_second=0.0,
                structured_output_reliable=False,
                rejection_reason=f"unsupported_optimization:{','.join(unsupported)}",
                notes=("candidate skipped because the optimization is not compatible with Ampere",),
            )

        size_billions = candidate.size_billions or _infer_model_size_billions(candidate.model) or 14.0
        bytes_per_param = _bytes_per_param(candidate.quantization)
        shards = machine.tensor_parallel_size if candidate.backend == "vllm" else machine.gpu_count
        weight_gb_per_gpu = (size_billions * bytes_per_param) / max(shards, 1)
        kv_gb_per_gpu = _estimate_kv_cache_gb(size_billions, candidate.context_size, candidate.concurrency, shards)
        overhead_gb = 2.2 if candidate.backend == "vllm" else 1.4
        optimization_delta = sum(
            float(_OPTIMIZATION_REGISTRY.get(name, {}).get("memory_delta_gb", 0.0))
            for name in candidate.optimizations
        )
        estimated_vram = round(max(weight_gb_per_gpu + kv_gb_per_gpu + overhead_gb + optimization_delta, 0.1), 3)

        headroom = machine.effective_vram_per_gpu_gb - estimated_vram
        if headroom < -0.5:
            status = "rejected"
            rejection_reason = "startup_oom"
        elif headroom < 1.5:
            status = "borderline"
            rejection_reason = None
        else:
            status = "deployable"
            rejection_reason = None

        estimated_ttft_ms = round(220.0 + (estimated_vram / machine.effective_vram_per_gpu_gb) * 900.0, 3)
        decode_base = 44.0 if candidate.backend == "vllm" else 34.0
        quant_bonus = {
            "fp8": 8.0,
            "awq-4bit": 6.0,
            "gptq-4bit": 5.0,
            "q4_k_m": 3.0,
            "q6_k": 1.0,
            "q8_0": -2.0,
        }.get(candidate.quantization, 0.0)
        decode_penalty = (candidate.context_size / 8192 - 1.0) * 4.0 + (candidate.concurrency - 1) * 3.0
        estimated_decode = round(max(decode_base + quant_bonus - decode_penalty, 4.0), 3)

        structured_output_reliable = status != "rejected" and candidate.family not in {"minimax"}
        notes = []
        if status == "borderline":
            notes.append("candidate is near the dual-3090 headroom limit and should be probed before longer runs")
        if "turboquant-3" in candidate.optimizations:
            notes.append("uses an experimental memory optimization path and should be verified on the target runtime")
        return ProbeResult(
            status=status,
            estimated_vram_per_gpu_gb=estimated_vram,
            estimated_ttft_ms=estimated_ttft_ms,
            estimated_decode_tokens_per_second=estimated_decode,
            structured_output_reliable=structured_output_reliable,
            rejection_reason=rejection_reason,
            notes=tuple(notes),
        )


class AutotuneOrchestrator:
    def __init__(
        self,
        recommender: LLMFitRecommender | None = None,
        prober: HeuristicProbe | None = None,
        runner: BenchmarkRunner | None = None,
    ) -> None:
        self._recommender = recommender or LLMFitRecommender()
        self._prober = prober or HeuristicProbe()
        self._runner = runner or BenchmarkRunner()

    def run(
        self,
        machine_id: str,
        strategy_id: str,
        *,
        live_config: LiveBenchmarkConfig | None = None,
        catalog_path: str | None = None,
    ) -> DeploymentRecommendation:
        machine = get_machine_profile(machine_id)
        strategy = get_search_strategy(strategy_id)
        candidates = self._recommender.recommend(machine, strategy)
        workloads = _build_autotune_workloads(catalog_path)
        evaluated: list[CandidateEvaluation] = []
        rejected: list[CandidateEvaluation] = []

        for index, candidate in enumerate(candidates):
            assigned = replace(candidate, metadata={**candidate.metadata, "port": _candidate_port(index, candidate.backend)})
            profile = build_profile(assigned, machine)
            probe = self._prober.probe(assigned, machine)
            live_failure_reason: str | None = None
            measured = None
            if probe.status != "rejected" and live_config and live_config.enabled:
                measured = self._run_live_benchmark(profile, workloads, live_config)
                if measured["status"] != "ok":
                    probe = replace(
                        probe,
                        status="rejected",
                        structured_output_reliable=False,
                        rejection_reason=str(measured["reason"]),
                        notes=tuple(list(probe.notes) + [str(measured["reason"])]),
                    )
                    live_failure_reason = str(measured["reason"])
            score = _score_candidate(assigned, probe, strategy)
            if measured and measured["status"] == "ok":
                score = _apply_measured_score(score, measured["results"], measured["summary"])
            evaluation = CandidateEvaluation(
                candidate=assigned,
                profile=profile,
                probe=probe,
                score=score,
                rank_reason=_rank_reason(assigned, probe, score, measured),
            )
            if probe.status == "rejected":
                if live_failure_reason:
                    evaluation = replace(
                        evaluation,
                        rank_reason=f"{evaluation.rank_reason}; live execution failed with {live_failure_reason}",
                    )
                rejected.append(evaluation)
                continue
            evaluated.append(evaluation)

        ranked = sorted(evaluated, key=lambda item: item.score, reverse=True)
        if not ranked:
            raise RuntimeError("No deployable or borderline candidates were found for the selected machine profile.")
        return DeploymentRecommendation(
            machine=machine,
            strategy=strategy,
            winner=ranked[0],
            runner_up=ranked[1] if len(ranked) > 1 else None,
            rejected=tuple(rejected),
            generated_at="local-autotune",
        )

    def _run_live_benchmark(
        self,
        profile: Profile,
        workloads: list[Workload],
        live_config: LiveBenchmarkConfig,
    ) -> dict[str, Any]:
        launcher = ServerLauncher(profile, live_config)
        try:
            launcher.start()
            results: list[BenchmarkResult] = []
            for workload in workloads:
                spec = BenchmarkRunSpec(
                    run_id=f"{profile.id}__{workload.id}__c1__r1",
                    profile_id=profile.id,
                    workload_id=workload.id,
                    concurrency=1,
                    repetition=1,
                )
                results.extend(self._runner.run_group(spec, profile, workload))
            summary = summarize_results(results)
            return {"status": "ok", "results": results, "summary": summary}
        except Exception as exc:
            return {"status": "failed", "reason": _classify_live_failure(exc)}
        finally:
            launcher.stop()


def build_profile(candidate: CandidateVariant, machine: MachineProfile) -> Profile:
    port = int(candidate.metadata.get("port", _candidate_port(0, candidate.backend)))
    api_base = f"http://127.0.0.1:{port}/v1"
    if candidate.backend == "vllm":
        args = [
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--model",
            candidate.model,
            "--tensor-parallel-size",
            str(machine.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(machine.gpu_memory_utilization),
            "--max-model-len",
            str(candidate.context_size),
        ]
        if "prefix-caching" in candidate.optimizations:
            args.append("--enable-prefix-caching")
        if "chunked-prefill" in candidate.optimizations:
            args.append("--enable-chunked-prefill")
        env = {}
        if "turboquant-3" in candidate.optimizations:
            env["BENCHLLM_EXPERIMENTAL_TURBOQUANT"] = "1"
        launch = LaunchSpec(
            command=["python", "-m", "vllm.entrypoints.openai.api_server"],
            args=args,
            env=env,
        )
    else:
        args = [
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--model",
            candidate.model,
            "--ctx-size",
            str(candidate.context_size),
            "--n-gpu-layers",
            "999",
            "--parallel",
            str(candidate.concurrency),
            "--split-mode",
            "row",
            "--tensor-split",
            "1,1",
        ]
        if "flash-attn" in candidate.optimizations:
            args.append("--flash-attn")
        if "q8-kv-cache" in candidate.optimizations:
            args.extend(["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"])
        launch = LaunchSpec(command=["./llama-server"], args=args, env={})
    return Profile(
        id=candidate.id,
        backend=candidate.backend,
        kind="inference",
        model=candidate.model,
        api_base=api_base,
        launch=launch,
        metadata={
            "family": candidate.family,
            "source": candidate.source,
            "quantization": candidate.quantization,
            "context_size": candidate.context_size,
            "concurrency": candidate.concurrency,
            "optimizations": list(candidate.optimizations),
        },
    )


def write_recommendation_bundle(recommendation: DeploymentRecommendation, output_dir: str | Path) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "recommendation.json").write_text(json.dumps(recommendation.to_dict(), indent=2), encoding="utf-8")
    launch_manifest = {
        "winner_id": recommendation.winner.candidate.id,
        "runner_up_id": recommendation.runner_up.candidate.id if recommendation.runner_up else None,
        "winner_endpoint": recommendation.winner.profile.api_base,
        "runner_up_endpoint": recommendation.runner_up.profile.api_base if recommendation.runner_up else None,
    }
    (root / "deployment-manifest.json").write_text(json.dumps(launch_manifest, indent=2), encoding="utf-8")
    return root


def load_recommendation(path: str | Path) -> DeploymentRecommendation:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    machine = MachineProfile(**raw["machine"])
    strategy = SearchStrategy(**raw["strategy"])
    winner = _load_evaluation(raw["winner"])
    runner_up = _load_evaluation(raw["runner_up"]) if raw.get("runner_up") else None
    rejected = tuple(_load_evaluation(item) for item in raw.get("rejected", []))
    return DeploymentRecommendation(
        machine=machine,
        strategy=strategy,
        winner=winner,
        runner_up=runner_up,
        rejected=rejected,
        generated_at=str(raw.get("generated_at", "loaded")),
    )


def _load_evaluation(raw: dict[str, Any]) -> CandidateEvaluation:
    candidate = CandidateVariant(**raw["candidate"])
    launch_raw = raw["profile"].get("launch")
    launch = LaunchSpec(**launch_raw) if launch_raw else None
    profile = Profile(
        id=raw["profile"]["id"],
        backend=raw["profile"]["backend"],
        kind=raw["profile"]["kind"],
        model=raw["profile"]["model"],
        api_base=raw["profile"]["api_base"],
        launch=launch,
        metadata=raw["profile"].get("metadata", {}),
    )
    probe = ProbeResult(**raw["probe"])
    return CandidateEvaluation(
        candidate=candidate,
        profile=profile,
        probe=probe,
        score=float(raw["score"]),
        rank_reason=raw["rank_reason"],
    )


def _score_candidate(candidate: CandidateVariant, probe: ProbeResult, strategy: SearchStrategy) -> float:
    if probe.status == "rejected":
        return -10_000.0
    score = 1000.0
    if probe.structured_output_reliable:
        score += 120.0
    score -= probe.estimated_ttft_ms * 0.35
    score += probe.estimated_decode_tokens_per_second * 7.0
    score += max(candidate.context_size / 1024 - 8.0, 0.0) * 2.0
    score += {"vllm": 40.0, "llama.cpp": 15.0}.get(candidate.backend, 0.0)
    score += {"fp8": 12.0, "awq-4bit": 16.0, "gptq-4bit": 12.0, "q4_k_m": 8.0, "q6_k": 5.0}.get(
        candidate.quantization,
        0.0,
    )
    if probe.status == "borderline":
        score -= 75.0
    if any(name in strategy.experimental_optimizations for name in candidate.optimizations):
        score += 6.0
    llmfit_score = candidate.metadata.get("llmfit_score")
    if isinstance(llmfit_score, (int, float)):
        score += float(llmfit_score) * 0.1
    return round(score, 3)


def _rank_reason(candidate: CandidateVariant, probe: ProbeResult, score: float, measured: dict[str, Any] | None = None) -> str:
    reason = (
        f"{candidate.backend} {candidate.quantization} scored {score} with "
        f"estimated TTFT {probe.estimated_ttft_ms} ms and decode "
        f"{probe.estimated_decode_tokens_per_second} tok/s"
    )
    if measured and measured["status"] == "ok":
        summaries = measured["summary"]
        if summaries:
            first = summaries[0]
            reason = (
                f"{candidate.backend} {candidate.quantization} scored {score} with "
                f"measured TTFT {first.median_ttft_ms} ms and decode "
                f"{first.median_decode_tokens_per_second} tok/s"
            )
    if probe.status == "borderline":
        return f"{reason}; kept as a fallback because it is near the VRAM limit"
    return reason


def _parse_llmfit_payload(payload: str, strategy: SearchStrategy) -> list[CandidateVariant]:
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError:
        return []
    items = _extract_candidate_items(raw)
    candidates = [_candidate_from_llmfit(item, strategy) for item in items]
    return [item for item in candidates if item is not None]


def _extract_candidate_items(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if not isinstance(raw, dict):
        return []
    for key in ("recommendations", "candidates", "models", "results"):
        value = raw.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    if any(key in raw for key in ("model", "model_id", "name")):
        return [raw]
    return []


def _candidate_from_llmfit(item: dict[str, Any], strategy: SearchStrategy) -> CandidateVariant | None:
    model = str(item.get("model") or item.get("model_id") or item.get("name") or "").strip()
    if not model:
        return None
    backend = str(item.get("runtime") or item.get("backend") or strategy.backend_order[0]).strip().lower()
    backend = "llama.cpp" if "llama" in backend else "vllm"
    quantization = str(item.get("quantization") or item.get("quant") or _infer_quantization(model))
    family = str(item.get("family") or _infer_family(model))
    size_billions = _extract_float(item, ("size_billions", "parameters_b", "params_b"))
    score = _extract_float(item, ("score", "total_score"))
    optimizations = ["prefix-caching", "chunked-prefill"] if backend == "vllm" else ["flash-attn", "q8-kv-cache"]
    if "ream" in model.lower():
        optimizations.append("ream")
    return CandidateVariant(
        id=_slugify(f"{family}-{backend}-{quantization}-{strategy.context_tiers[0]}"),
        family=family,
        model=model,
        backend=backend,
        quantization=quantization,
        context_size=int(item.get("context") or item.get("context_size") or strategy.context_tiers[0]),
        concurrency=strategy.concurrency_tiers[1],
        source="llmfit",
        size_billions=size_billions,
        optimizations=tuple(optimizations),
        metadata={"llmfit_score": score} if score is not None else {},
    )


def _curated_candidates(machine: MachineProfile, strategy: SearchStrategy) -> list[CandidateVariant]:
    del machine
    base_context = strategy.context_tiers[0]
    concurrency = strategy.concurrency_tiers[1]
    return [
        CandidateVariant(
            id="qwen3-coder-next-vllm-awq",
            family="qwen3-coder-next",
            model="cyankiwi/Qwen3-Coder-Next-REAM-AWQ-4bit",
            backend="vllm",
            quantization="awq-4bit",
            context_size=base_context,
            concurrency=concurrency,
            source="curated",
            size_billions=32.0,
            optimizations=("prefix-caching", "chunked-prefill", "ream", "turboquant-3"),
        ),
        CandidateVariant(
            id="devstral-small-vllm-fp8",
            family="devstral-small",
            model="mistralai/Devstral-Small-2507",
            backend="vllm",
            quantization="fp8",
            context_size=base_context,
            concurrency=concurrency,
            source="curated",
            size_billions=24.0,
            optimizations=("prefix-caching", "chunked-prefill"),
        ),
        CandidateVariant(
            id="qwen35-moe-vllm-awq",
            family="qwen3.5-moe",
            model="Qwen/Qwen3.5-35B-A3B-AWQ",
            backend="vllm",
            quantization="awq-4bit",
            context_size=base_context,
            concurrency=1,
            source="curated",
            size_billions=35.0,
            optimizations=("prefix-caching", "chunked-prefill"),
        ),
        CandidateVariant(
            id="devstral-small-llamacpp-q4km",
            family="devstral-small",
            model="/models/devstral-small-2507-q4_k_m.gguf",
            backend="llama.cpp",
            quantization="q4_k_m",
            context_size=base_context,
            concurrency=concurrency,
            source="curated",
            size_billions=24.0,
            remote=False,
            optimizations=("flash-attn", "q8-kv-cache"),
        ),
    ]


def _is_optimization_supported(name: str, architecture: str) -> bool:
    optimization = _OPTIMIZATION_REGISTRY.get(name)
    if optimization is None:
        return True
    return architecture in {"ampere", "hopper"} and optimization["min_arch"] in {"ampere", architecture}


def _estimate_kv_cache_gb(size_billions: float, context_size: int, concurrency: int, shards: int) -> float:
    return round((size_billions * 0.08) * (context_size / 8192.0) * max(concurrency, 1) / max(shards, 1), 3)


def _infer_model_size_billions(model: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*[Bb]", model)
    if not match:
        return None
    return float(match.group(1))


def _infer_quantization(model: str) -> str:
    text = model.lower()
    if "awq" in text:
        return "awq-4bit"
    if "gptq" in text:
        return "gptq-4bit"
    if "fp8" in text:
        return "fp8"
    if "q4_k_m" in text:
        return "q4_k_m"
    if "q6" in text:
        return "q6_k"
    return "fp16"


def _infer_family(model: str) -> str:
    normalized = model.split("/")[-1].lower()
    for family in ("qwen3-coder-next", "devstral-small", "qwen3.5", "minimax"):
        if family in normalized:
            return family
    return normalized.split("-")[0]


def _bytes_per_param(quantization: str) -> float:
    return {
        "fp16": 2.0,
        "fp8": 1.0,
        "awq-4bit": 0.6,
        "gptq-4bit": 0.65,
        "q4_k_m": 0.55,
        "q6_k": 0.8,
        "q8_0": 1.0,
    }.get(quantization, 1.2)


def _extract_float(item: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _candidate_port(index: int, backend: str) -> int:
    return (8000 if backend == "vllm" else 9000) + index


def _build_autotune_workloads(catalog_path: str | None) -> list[Workload]:
    if catalog_path:
        catalog = load_catalog(catalog_path)
        selected_ids = ("json-small", "code-plan")
        available = [catalog.workloads[workload_id] for workload_id in selected_ids if workload_id in catalog.workloads]
        if available:
            return available
    return [
        Workload(
            id="json-small",
            request={
                "temperature": 0,
                "max_tokens": 160,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": "Return strict JSON only, no prose."},
                    {
                        "role": "user",
                        "content": 'Return {"status":"ok","changed_files":["benchllm/cli.py"],"risk":"low"}',
                    },
                ],
            },
            validations=ValidationRules(expect_json=True),
        ),
        Workload(
            id="code-plan",
            request={
                "temperature": 0,
                "max_tokens": 192,
                "messages": [
                    {"role": "system", "content": "You are a precise coding assistant."},
                    {"role": "user", "content": "Propose a concise refactor plan for a Python CLI."},
                ],
            },
        ),
    ]


def _apply_measured_score(score: float, results: list[BenchmarkResult], summary: list[Any]) -> float:
    del summary
    if not results:
        return score
    validation_rate = sum(1 for item in results if item.validation_passed) / len(results)
    median_ttft = sorted(item.ttft_ms for item in results)[len(results) // 2]
    median_decode = sorted(item.decode_tokens_per_second for item in results)[len(results) // 2]
    score += validation_rate * 180.0
    score -= median_ttft * 0.08
    score += median_decode * 5.0
    return round(score, 3)


def _classify_live_failure(exc: Exception) -> str:
    message = str(exc).lower()
    if "timed out" in message or "timeout" in message:
        return "startup_timeout"
    if "connection refused" in message or "connecterror" in message:
        return "server_unreachable"
    if "json" in message:
        return "structured_output_invalid"
    if "404" in message or "405" in message:
        return "api_incompatible"
    return f"runtime_failure:{exc.__class__.__name__}"


class ServerLauncher:
    def __init__(self, profile: Profile, live_config: LiveBenchmarkConfig) -> None:
        self._profile = profile
        self._live_config = live_config
        self._process: subprocess.Popen[bytes] | None = None
        self._log_handle: Any = None

    def start(self) -> None:
        if self._profile.launch is None:
            raise RuntimeError("missing_launch_spec")
        env = os.environ.copy()
        env.update(self._profile.launch.env)
        command = [*self._resolve_command(), *self._profile.launch.args]
        log_path = None
        if self._live_config.log_dir:
            log_dir = Path(self._live_config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{self._profile.id}.log"
            self._log_handle = log_path.open("ab")
        stdout = self._log_handle or subprocess.DEVNULL
        self._process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            env=env,
        )
        self._wait_until_ready()

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        self._process = None

    def _resolve_command(self) -> list[str]:
        command = list(self._profile.launch.command)
        if command[:3] == ["python", "-m", "vllm.entrypoints.openai.api_server"]:
            python_bin = os.environ.get("VLLM_PYTHON_BIN")
            if python_bin:
                return [python_bin, "-m", "vllm.entrypoints.openai.api_server"]
            stack_root = os.environ.get("BENCHLLM_STACK_ROOT")
            if stack_root:
                default_python = Path(stack_root) / ".venvs/vllm/bin/python"
                if default_python.exists():
                    return [str(default_python), "-m", "vllm.entrypoints.openai.api_server"]
        if self._profile.backend == "llama.cpp" and command == ["./llama-server"]:
            root = os.environ.get("LLAMA_CPP_ROOT")
            if root:
                candidate = Path(root) / "build/bin/llama-server"
                if candidate.exists():
                    return [str(candidate)]
            stack_root = os.environ.get("BENCHLLM_STACK_ROOT")
            if stack_root:
                candidate = Path(stack_root) / "src/llama.cpp/build/bin/llama-server"
                if candidate.exists():
                    return [str(candidate)]
        return command

    def _wait_until_ready(self) -> None:
        timeout_at = time.time() + self._live_config.startup_timeout_seconds
        models_url = f"{self._profile.api_base.rstrip('/')}/models"
        with httpx.Client(timeout=httpx.Timeout(self._live_config.warmup_timeout_seconds, connect=2.0)) as client:
            while time.time() < timeout_at:
                if self._process is not None and self._process.poll() is not None:
                    raise RuntimeError(f"server_exited:{self._process.returncode}")
                try:
                    response = client.get(models_url)
                    if response.status_code < 500:
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(1.0)
        if self._process is not None:
            try:
                os.kill(self._process.pid, signal.SIGTERM)
            except OSError:
                pass
        raise TimeoutError(f"Timed out waiting for {self._profile.id} to become ready")
