"""Checkpoint loading from a local directory or the Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from transformers.initialization import no_init_weights

from ..config import HaruConfig
from .haru import HaruModel

DEFAULT_REPO = "higurashi-ai/Haru-4B"
_FILES = ("config.json", "model.safetensors", "tokenizer.json")


def resolve_checkpoint(path_or_repo: str = DEFAULT_REPO, revision: str | None = None) -> Path:
    """Directory holding ``config.json``, ``model.safetensors`` and ``tokenizer.json``."""
    path = Path(path_or_repo).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(path_or_repo, revision=revision, allow_patterns=list(_FILES)))


def load_tokenizer(path_or_repo: str = DEFAULT_REPO) -> Tokenizer:
    return Tokenizer.from_file(str(resolve_checkpoint(path_or_repo) / "tokenizer.json"))


def load_model(
    path_or_repo: str = DEFAULT_REPO,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    revision: str | None = None,
) -> HaruModel:
    """Build a :class:`HaruModel` in eval mode with its tokenizer attached.

    ``device`` defaults to CUDA when available. ``dtype`` defaults to the
    checkpoint's (bfloat16); pass ``torch.float32`` to run on CPU.
    """
    directory = resolve_checkpoint(path_or_repo, revision)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = HaruConfig.from_file(directory / "config.json")
    with no_init_weights():
        model = HaruModel(config)
    state = load_file(str(directory / "model.safetensors"), device=str(device))
    model.load_state_dict(state, strict=True, assign=True)
    model.to(device=device, dtype=dtype)
    model.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    return model.eval().requires_grad_(False)
