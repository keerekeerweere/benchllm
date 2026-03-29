import contextlib
import io
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from benchllm.autotune import (
    AutotuneOrchestrator,
    CandidateVariant,
    HeuristicProbe,
    LLMFitRecommender,
    LiveBenchmarkConfig,
    get_machine_profile,
    get_search_strategy,
    load_recommendation,
)
from benchllm.catalog import LaunchSpec, Profile, ValidationRules, Workload, BenchmarkRunSpec
from benchllm.runner import BenchmarkResult
from benchllm.cli import main


class StaticRecommender(LLMFitRecommender):
    def __init__(self, candidates: list[CandidateVariant]) -> None:
        self._candidates = candidates

    def recommend(self, machine, strategy):  # type: ignore[override]
        del machine, strategy
        return list(self._candidates)


class AutotuneTest(unittest.TestCase):
    def test_probe_rejects_oversized_fp16_candidate(self) -> None:
        machine = get_machine_profile("dual-3090")
        probe = HeuristicProbe()
        candidate = CandidateVariant(
            id="too-large",
            family="oversized",
            model="Example/70B-FP16",
            backend="vllm",
            quantization="fp16",
            context_size=16384,
            concurrency=2,
            source="test",
            size_billions=70.0,
            optimizations=("prefix-caching",),
        )

        result = probe.probe(candidate, machine)

        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.rejection_reason, "startup_oom")

    def test_orchestrator_prefers_deployable_candidate_over_rejected_one(self) -> None:
        orchestrator = AutotuneOrchestrator(
            recommender=StaticRecommender(
                [
                    CandidateVariant(
                        id="too-large",
                        family="oversized",
                        model="Example/70B-FP16",
                        backend="vllm",
                        quantization="fp16",
                        context_size=16384,
                        concurrency=2,
                        source="test",
                        size_billions=70.0,
                        optimizations=("prefix-caching",),
                    ),
                    CandidateVariant(
                        id="good-fit",
                        family="devstral-small",
                        model="mistralai/Devstral-Small-2507",
                        backend="vllm",
                        quantization="fp8",
                        context_size=8192,
                        concurrency=2,
                        source="test",
                        size_billions=24.0,
                        optimizations=("prefix-caching", "chunked-prefill"),
                    ),
                    CandidateVariant(
                        id="fallback",
                        family="devstral-small",
                        model="/models/devstral-small-2507-q4_k_m.gguf",
                        backend="llama.cpp",
                        quantization="q4_k_m",
                        context_size=8192,
                        concurrency=2,
                        source="test",
                        size_billions=24.0,
                        remote=False,
                        optimizations=("flash-attn", "q8-kv-cache"),
                    ),
                ]
            )
        )

        recommendation = orchestrator.run("dual-3090", "fast-agentic")

        self.assertEqual(recommendation.winner.candidate.id, "good-fit")
        self.assertEqual(recommendation.runner_up.candidate.id, "fallback")
        self.assertEqual(recommendation.rejected[0].candidate.id, "too-large")

    def test_cli_autotune_writes_bundle_and_recommend_reads_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "autotune"

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["autotune", "--output-dir", str(output_dir), "--heuristic-only"])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "recommendation.json").exists())
            self.assertTrue((output_dir / "deployment-manifest.json").exists())
            self.assertTrue((output_dir / "launchers").exists())
            launcher = (output_dir / "launchers" / "qwen3-coder-next-vllm-awq.sh").read_text(encoding="utf-8")
            self.assertIn("BENCHLLM_EXPERIMENTAL_TURBOQUANT=1", launcher)

            recommendation = load_recommendation(output_dir / "recommendation.json")
            self.assertEqual(recommendation.winner.candidate.id, "qwen3-coder-next-vllm-awq")

            recommend_stdout = io.StringIO()
            with contextlib.redirect_stdout(recommend_stdout):
                exit_code = main(["recommend", "--results", str(output_dir / "recommendation.json")])

            self.assertEqual(exit_code, 0)
            self.assertIn("winner=cyankiwi/Qwen3-Coder-Next-REAM-AWQ-4bit", recommend_stdout.getvalue())

    def test_live_benchmark_path_runs_with_launcher_and_runner(self) -> None:
        profile = Profile(
            id="toy-server",
            backend="custom",
            kind="inference",
            model="toy/model",
            api_base="http://127.0.0.1:8011/v1",
            launch=LaunchSpec(command=["python"], args=["-m", "toy"], env={}),
            metadata={},
        )
        workloads = [
            Workload(
                id="json-small",
                request={
                    "temperature": 0,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "Return JSON"}],
                },
                validations=ValidationRules(expect_json=True),
            )
        ]
        fake_result = BenchmarkResult(
            run_id="toy-server__json-small__c1__r1",
            profile_id="toy-server",
            workload_id="json-small",
            worker_index=0,
            status_code=200,
            ttft_ms=120.0,
            total_duration_ms=300.0,
            prompt_tokens=12,
            completion_tokens=4,
            total_tokens=16,
            decode_tokens_per_second=22.0,
            validation_passed=True,
            validation_error=None,
            response_text='{"status":"ok"}',
        )
        fake_runner = mock.Mock()
        fake_runner.run_group.return_value = [fake_result]
        orchestrator = AutotuneOrchestrator(recommender=StaticRecommender([]), runner=fake_runner)

        with mock.patch("benchllm.autotune.ServerLauncher") as launcher_cls:
            launcher = launcher_cls.return_value
            measured = orchestrator._run_live_benchmark(
                profile,
                workloads,
                LiveBenchmarkConfig(enabled=True, startup_timeout_seconds=10.0, warmup_timeout_seconds=5.0),
            )

        self.assertEqual(measured["status"], "ok")
        self.assertEqual(len(measured["results"]), 1)
        self.assertTrue(measured["results"][0].validation_passed)
        launcher.start.assert_called_once()
        launcher.stop.assert_called_once()
        fake_runner.run_group.assert_called_once()

    def test_llmfit_payload_parser_uses_context_and_score_fields(self) -> None:
        strategy = get_search_strategy("fast-agentic")
        payload = """
        {
          "recommendations": [
            {
              "model": "Qwen/Qwen3-Coder-Next-FP8",
              "runtime": "vllm",
              "score": 91.5,
              "context_size": 16384,
              "size_billions": 32
            }
          ]
        }
        """

        from benchllm.autotune import _parse_llmfit_payload

        candidates = _parse_llmfit_payload(payload, strategy)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].context_size, 16384)
        self.assertEqual(candidates[0].metadata["llmfit_score"], 91.5)


if __name__ == "__main__":
    unittest.main()
