import tempfile
import textwrap
import unittest
from pathlib import Path

from benchllm.catalog import build_run_matrix, load_catalog


CATALOG_YAML = textwrap.dedent(
    """
    defaults:
      api_base: http://localhost:8000/v1
      context_size: 8192
    profiles:
      - id: vllm-qwen3-coder-next-fp8
        backend: vllm
        kind: inference
        model: Qwen/Qwen3-Coder-Next-FP8
        launch:
          command:
            - python
            - -m
            - vllm.entrypoints.openai.api_server
          args:
            - --tensor-parallel-size
            - "2"
      - id: llama-devstral-q4
        backend: llama.cpp
        kind: inference
        model: devstral-small-2507-q4_k_m.gguf
        api_base: http://localhost:8080/v1
        launch:
          command:
            - ./llama-server
          args:
            - --flash-attn
    workloads:
      - id: json-small
        request:
          temperature: 0
          max_tokens: 128
          response_format:
            type: json_object
          messages:
            - role: system
              content: Return valid JSON only.
            - role: user
              content: Return {"status":"ok","count":1}
        validations:
          expect_json: true
      - id: code-plan
        request:
          temperature: 0
          max_tokens: 256
          messages:
            - role: user
              content: Outline a refactor plan for a Python CLI.
    matrix:
      profiles:
        - vllm-qwen3-coder-next-fp8
      workloads:
        - json-small
        - code-plan
      concurrencies:
        - 1
        - 2
      repetitions: 2
    """
)


class CatalogTest(unittest.TestCase):
    def test_loads_catalog_and_applies_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "catalog.yaml"
            path.write_text(CATALOG_YAML, encoding="utf-8")

            catalog = load_catalog(path)

        self.assertEqual(catalog.defaults.api_base, "http://localhost:8000/v1")
        self.assertEqual(catalog.defaults.context_size, 8192)
        self.assertEqual(len(catalog.profiles), 2)
        self.assertEqual(catalog.profiles["vllm-qwen3-coder-next-fp8"].api_base, "http://localhost:8000/v1")
        self.assertEqual(catalog.profiles["llama-devstral-q4"].api_base, "http://localhost:8080/v1")

    def test_builds_run_matrix_from_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "catalog.yaml"
            path.write_text(CATALOG_YAML, encoding="utf-8")

            catalog = load_catalog(path)
            runs = build_run_matrix(catalog)

        self.assertEqual(len(runs), 8)
        first = runs[0]
        self.assertEqual(first.profile_id, "vllm-qwen3-coder-next-fp8")
        self.assertEqual(first.workload_id, "json-small")
        self.assertEqual(first.concurrency, 1)
        self.assertEqual(first.repetition, 1)
        self.assertTrue(first.run_id.startswith("vllm-qwen3-coder-next-fp8__json-small__c1__r1"))

    def test_real_catalog_uses_ream_awq_for_default_qwen3_coder_next_profile(self) -> None:
        catalog = load_catalog("/home/dbram/work/benchllm/catalogs/dual-3090-openai.yaml")

        profile = catalog.profiles["vllm-qwen3-coder-next-awq"]

        self.assertEqual(profile.model, "cyankiwi/Qwen3-Coder-Next-REAM-AWQ-4bit")
        self.assertIn("--model", profile.launch.args)
        self.assertIn("cyankiwi/Qwen3-Coder-Next-REAM-AWQ-4bit", profile.launch.args)
        self.assertIn("vllm-qwen3-coder-next-awq", catalog.matrix.profiles)
        self.assertNotIn("vllm-qwen3-coder-next-fp8", catalog.matrix.profiles)


if __name__ == "__main__":
    unittest.main()
