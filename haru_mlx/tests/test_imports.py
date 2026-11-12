"""The package depends only on what it declares, and its pure modules need no MLX."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parent.parent / "haru_mlx"
FORBIDDEN = {"torch", "transformers", "mlx_lm", "haru"}
ALLOWED = {
    "mlx", "numpy", "tokenizers", "huggingface_hub", "scipy", "soundfile",  # core
    "aiohttp", "sphn",  # server
    "aiortc", "av",  # extra: webrtc
    "sounddevice",  # extra: local
}
# Modules that must import without MLX (they are what the MLX-free tests use).
PURE = ["", "audio", "backends", "config", "frames", "hub", "lm_config", "playout", "prompt", "sampling", "weights", "worker"]


def module_names(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_only_declared_dependencies_are_used():
    files = sorted(PACKAGE.rglob("*.py"))
    assert files
    for path in files:
        third_party = {name for name in module_names(path) if name not in sys.stdlib_module_names}
        assert not third_party & FORBIDDEN, f"{path.name} imports {sorted(third_party & FORBIDDEN)}"
        assert third_party <= ALLOWED, f"{path.name} imports undeclared {sorted(third_party - ALLOWED)}"


@pytest.mark.parametrize("module", PURE)
def test_pure_modules_import_without_mlx(module):
    name = f"haru_mlx.{module}" if module else "haru_mlx"
    # A None entry in sys.modules makes "import mlx" raise ImportError.
    code = f"import sys; sys.modules['mlx'] = None; import {name}; assert 'mlx.core' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=PACKAGE.parent)
    assert result.returncode == 0, result.stderr


def test_server_modules_import_without_mlx():
    pytest.importorskip("aiohttp")
    pytest.importorskip("sphn")
    code = "import sys; sys.modules['mlx'] = None; import haru_mlx.server, haru_mlx.native, haru_mlx.run_inference"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=PACKAGE.parent)
    assert result.returncode == 0, result.stderr
