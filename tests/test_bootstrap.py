import subprocess
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class BootstrapScriptTest(unittest.TestCase):
    def _make_minimal_path_bin(self, tmpdir: str) -> str:
        bin_dir = Path(tmpdir) / "bin"
        bin_dir.mkdir()
        for command in ("git", "cmake", "python3"):
            target = Path("/usr/bin") / command
            link = bin_dir / command
            if target.exists():
                link.symlink_to(target)
        return str(bin_dir)

    def test_install_sh_dry_run_uses_pipx_for_uv_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "stack"
            bin_dir = Path(tmpdir) / "bin"
            bin_dir.mkdir()
            pipx_path = bin_dir / "pipx"
            pipx_path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            pipx_path.chmod(0o755)
            proc = subprocess.run(
                ["bash", "install.sh"],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
                env={
                    "PATH": f"{bin_dir}:/usr/bin:/bin",
                    "INSTALL_DRY_RUN": "1",
                    "BENCHLLM_INSTALL_ROOT": str(root),
                },
            )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("pipx install uv", proc.stdout)
        self.assertIn("--python-tool uv", proc.stdout)

    def test_install_sh_dry_run_uses_repo_default_without_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "stack"
            home = Path(tmpdir) / "home"
            home.mkdir()
            path_bin = self._make_minimal_path_bin(tmpdir)
            proc = subprocess.run(
                ["/bin/bash", "install.sh"],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
                env={
                    "PATH": path_bin,
                    "HOME": str(home),
                    "INSTALL_DRY_RUN": "1",
                    "BENCHLLM_INSTALL_ROOT": str(root),
                },
            )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("git clone https://github.com/keerekeerweere/benchllm", proc.stdout)
        self.assertIn("bash", proc.stdout)
        self.assertIn("bash /tmp", proc.stdout)
        self.assertIn("/run.sh --python-tool venv --dry-run", proc.stdout)
        self.assertIn("cat > ", proc.stdout)
        self.assertIn('/run.sh" <<\'EOF\'', proc.stdout)

    def test_install_sh_dry_run_falls_back_to_venv_when_uv_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "stack"
            home = Path(tmpdir) / "home"
            home.mkdir()
            path_bin = self._make_minimal_path_bin(tmpdir)
            proc = subprocess.run(
                ["/bin/bash", "install.sh"],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
                env={
                    "PATH": path_bin,
                    "HOME": str(home),
                    "INSTALL_DRY_RUN": "1",
                    "BENCHLLM_INSTALL_ROOT": str(root),
                },
            )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--python-tool venv", proc.stdout)

    def test_run_sh_dry_run_supports_uv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "stack"
            bin_dir = Path(tmpdir) / "bin"
            bin_dir.mkdir()
            uv_path = bin_dir / "uv"
            uv_path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            uv_path.chmod(0o755)
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
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
                env={
                    "PATH": f"{bin_dir}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                    "BENCHLLM_CUDACXX": "/usr/local/cuda/bin/nvcc",
                },
            )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("uv python install 3.12", proc.stdout)
        self.assertIn("uv venv", proc.stdout)
        self.assertNotIn(f"git clone {REPO_ROOT} {root}/src/benchllm", proc.stdout)
        self.assertNotIn("git clone https://github.com/vllm-project/vllm.git", proc.stdout)
        self.assertIn("git clone https://github.com/ggml-org/llama.cpp.git", proc.stdout)
        self.assertIn("uv pip install --python", proc.stdout)
        self.assertIn(" vllm", proc.stdout)
        self.assertIn(f" -e {REPO_ROOT}", proc.stdout)
        self.assertIn("python -m benchllm autotune", proc.stdout)
        self.assertIn("--machine dual-3090", proc.stdout)
        self.assertIn("--strategy fast-agentic", proc.stdout)

    def test_run_sh_dry_run_passes_explicit_cuda_compiler_to_cmake(self) -> None:
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
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
                env={
                    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                    "BENCHLLM_CUDACXX": "/usr/local/cuda/bin/nvcc",
                },
            )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc", proc.stdout)

    def test_run_sh_dry_run_falls_back_to_venv_when_uv_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "stack"
            proc = subprocess.run(
                [
                    "bash",
                    "run.sh",
                    "--dry-run",
                    "--root",
                    str(root),
                ],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
                env={
                    "PATH": "/usr/bin:/bin",
                    "BENCHLLM_CUDACXX": "/usr/local/cuda/bin/nvcc",
                },
            )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("python3 -m venv", proc.stdout)
        self.assertNotIn("uv venv", proc.stdout)

    def test_run_sh_prepare_only_uses_static_prepare_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "stack"
            bin_dir = Path(tmpdir) / "bin"
            bin_dir.mkdir()
            uv_path = bin_dir / "uv"
            uv_path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            uv_path.chmod(0o755)
            proc = subprocess.run(
                [
                    "bash",
                    "run.sh",
                    "--dry-run",
                    "--root",
                    str(root),
                    "--python-tool",
                    "uv",
                    "--prepare-only",
                ],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
                env={
                    "PATH": f"{bin_dir}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                    "BENCHLLM_CUDACXX": "/usr/local/cuda/bin/nvcc",
                },
            )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("python -m benchllm prepare", proc.stdout)
        self.assertNotIn("python -m benchllm autotune", proc.stdout)

    def test_root_env_example_exists_and_gitignore_ignores_env(self) -> None:
        repo_root = REPO_ROOT
        env_example = (repo_root / ".env.example").read_text(encoding="utf-8")
        gitignore = (repo_root / ".gitignore").read_text(encoding="utf-8")

        self.assertIn("HF_TOKEN=", env_example)
        self.assertIn("CUDA_VISIBLE_DEVICES=", env_example)
        self.assertIn(".env", gitignore)


if __name__ == "__main__":
    unittest.main()
