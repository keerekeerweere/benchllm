import subprocess
import tempfile
import unittest
from pathlib import Path


class BootstrapScriptTest(unittest.TestCase):
    def test_run_sh_dry_run_supports_uv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "stack"
            proc = subprocess.run(
                [
                    "bash",
                    "run.sh",
                    "--dry-run",
                    "--root",
                    str(root),
                    "--python-tool",
                    "uv",
                ],
                cwd="/home/dbram/work/benchllm",
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("uv venv", proc.stdout)
        self.assertIn("git clone https://github.com/vllm-project/vllm.git", proc.stdout)
        self.assertIn("git clone https://github.com/ggml-org/llama.cpp.git", proc.stdout)
        self.assertIn("python -m benchllm prepare", proc.stdout)

    def test_root_env_example_exists_and_gitignore_ignores_env(self) -> None:
        repo_root = Path("/home/dbram/work/benchllm")
        env_example = (repo_root / ".env.example").read_text(encoding="utf-8")
        gitignore = (repo_root / ".gitignore").read_text(encoding="utf-8")

        self.assertIn("HF_TOKEN=", env_example)
        self.assertIn("CUDA_VISIBLE_DEVICES=", env_example)
        self.assertIn(".env", gitignore)


if __name__ == "__main__":
    unittest.main()
