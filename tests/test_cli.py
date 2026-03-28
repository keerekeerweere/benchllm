import contextlib
import io
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from benchllm.cli import main


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
          command: [python, -m, vllm.entrypoints.openai.api_server]
          args:
            - --model
            - Qwen/Qwen3-Coder-Next-FP8
            - --port
            - "8000"
    workloads:
      - id: json-small
        request:
          temperature: 0
          max_tokens: 128
          messages:
            - role: user
              content: Return {"status":"ok"}
        validations:
          expect_json: true
    matrix:
      profiles: [vllm-qwen3-coder-next-fp8]
      workloads: [json-small]
      concurrencies: [1]
      repetitions: 1
    """
)


class CliTest(unittest.TestCase):
    def test_plan_command_prints_matrix_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "catalog.yaml"
            path.write_text(CATALOG_YAML, encoding="utf-8")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["plan", "--catalog", str(path)])

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("vllm-qwen3-coder-next-fp8__json-small__c1__r1", output)
        self.assertIn("Qwen/Qwen3-Coder-Next-FP8", output)

    def test_module_execution_prints_plan_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "catalog.yaml"
            path.write_text(CATALOG_YAML, encoding="utf-8")

            proc = subprocess.run(
                [sys.executable, "-m", "benchllm.cli", "plan", "--catalog", str(path)],
                cwd="/home/dbram/work/benchllm",
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(proc.returncode, 0)
        self.assertIn("vllm-qwen3-coder-next-fp8__json-small__c1__r1", proc.stdout)

    def test_prepare_command_writes_launch_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "catalog.yaml"
            output_dir = Path(tmpdir) / "runtime"
            path.write_text(CATALOG_YAML, encoding="utf-8")

            exit_code = main(["prepare", "--catalog", str(path), "--output-dir", str(output_dir)])

            self.assertEqual(exit_code, 0)
            launcher = output_dir / "launchers" / "vllm-qwen3-coder-next-fp8.sh"
            self.assertTrue(launcher.exists())
            contents = launcher.read_text(encoding="utf-8")
            self.assertIn("vllm.entrypoints.openai.api_server", contents)
            self.assertIn("Qwen/Qwen3-Coder-Next-FP8", contents)
            self.assertTrue(os.access(launcher, os.X_OK))
            env_script = output_dir / "env.sh"
            env_contents = env_script.read_text(encoding="utf-8")
            self.assertIn('if [ -f "$ROOT_DIR/.env" ]', env_contents)
            self.assertIn("HF_TOKEN", env_contents)
            env_example = output_dir / ".env.example"
            self.assertTrue(env_example.exists())
            self.assertIn("HF_TOKEN=", env_example.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
