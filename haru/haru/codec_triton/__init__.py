"""Optional fast GPU backend for the VibeVoice codec: Triton kernels replayed as CUDA graphs.

``TritonCodecStream`` is a drop-in for ``haru.models.vibevoice_codec.CodecStream``.
It needs the ``triton`` package and a CUDA device; check ``is_available()`` and
fall back to ``CodecStream`` otherwise::

    from haru.codec_triton import TritonCodecStream, is_available

    stream = TritonCodecStream(codec, batch) if is_available() else CodecStream(codec, batch)
"""

from __future__ import annotations

from importlib.util import find_spec

import torch

__all__ = ["TritonCodecStream", "is_available"]


def is_available() -> bool:
    """Whether the backend can run here: ``triton`` is installed and an NVIDIA CUDA device is usable."""
    return find_spec("triton") is not None and torch.version.cuda is not None and torch.cuda.is_available()


def __getattr__(name: str):
    if name == "TritonCodecStream":  # loaded on first use, so that this package can be probed without triton
        from .stream import TritonCodecStream

        return TritonCodecStream
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
