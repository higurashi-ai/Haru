"""``haru_mlx`` shares its framework-independent code with the PyTorch package ``haru``.

Compared on source text, so neither package is loaded (``haru`` needs
PyTorch). Skipped when ``haru`` is not installed next to ``haru_mlx``.
"""

import ast
from importlib.util import find_spec
from pathlib import Path

import pytest

MLX = Path(__file__).parent.parent / "haru_mlx"


@pytest.fixture(scope="module")
def torch_package() -> Path:
    try:
        spec = find_spec("haru")
    except (ImportError, ValueError):
        spec = None
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("the PyTorch package haru is not installed")
    return Path(list(spec.submodule_search_locations)[0])


def definitions(path: Path) -> dict[str, str]:
    """``{name: normalized source}`` of the top-level definitions and constants of a module."""
    out = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = ast.dump(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0]
            for name in [target] if isinstance(target, ast.Name) else getattr(target, "elts", []):
                out[name.id] = ast.dump(node)
    return out


def methods(path: Path, cls: str) -> dict[str, str]:
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return {item.name: ast.dump(item) for item in node.body if isinstance(item, ast.FunctionDef)}
    raise KeyError(cls)


@pytest.mark.parametrize("name", ["audio.py", "config.py"])
def test_identical_files(torch_package, name):
    assert (MLX / name).read_bytes() == (torch_package / name).read_bytes()


@pytest.mark.parametrize(
    "ours, theirs, names",
    [
        ("sampling.py", "sampling.py", ["DryConfig", "dry_penalties", "StreamDetokenizer"]),
        (
            "prompt.py", "prompt.py",
            ["CARD_FIELDS", "render_card", "system_prompt_ids", "reference_ids", "reference_frames", "prompt_length"],
        ),
        ("frames.py", "session.py", ["Frame"]),
        ("backends.py", "backends.py", ["SessionSettings", "Conversation", "Backend"]),
        ("hub.py", "models/loaders.py", ["DEFAULT_REPO", "_FILES", "resolve_checkpoint"]),
    ],
)
def test_shared_definitions(torch_package, ours, theirs, names):
    mine, reference = definitions(MLX / ours), definitions(torch_package / theirs)
    for name in names:
        assert mine[name].replace("haru_mlx", "haru") == reference[name], name


def fields(path: Path, cls: str) -> dict[str, str]:
    """``{field: "annotation = default"}`` of a dataclass."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return {
                item.target.id: f"{ast.unparse(item.annotation)} = {ast.unparse(item.value)}"
                for item in node.body
                if isinstance(item, ast.AnnAssign)
            }
    raise KeyError(cls)


def test_duplex_config_has_the_same_fields_and_defaults(torch_package):
    reference = fields(torch_package / "session.py", "DuplexConfig")
    reference.pop("codec_backend", None)  # selects between the PyTorch codec implementations
    assert fields(MLX / "frames.py", "DuplexConfig") == reference


def test_dry_state_bookkeeping_is_shared(torch_package):
    mine, reference = methods(MLX / "sampling.py", "DryState"), methods(torch_package / "sampling.py", "DryState")
    assert mine["__init__"] == reference["__init__"] and mine["update"] == reference["update"]
