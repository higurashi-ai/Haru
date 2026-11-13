"""Triton codec backend: availability probing, and parity with the PyTorch ``CodecStream`` on a real checkpoint
(``HARU_MODEL=/path/to/Haru-4B pytest -m gpu tests/test_codec_triton.py``)."""

import os
import threading
from importlib.util import find_spec

import pytest
import torch

from haru import codec_triton

FRAME = 3200


def gpu(test):
    """Mark a test that needs a checkpoint, triton and a CUDA device."""
    for mark in (
        pytest.mark.skipif(not codec_triton.is_available(), reason="needs triton and a CUDA device"),
        pytest.mark.skipif(not os.environ.get("HARU_MODEL"), reason="set HARU_MODEL to a checkpoint"),
        pytest.mark.gpu,
    ):
        test = mark(test)
    return test


def test_is_available_reports_missing_requirements(monkeypatch):
    assert isinstance(codec_triton.is_available(), bool)
    assert codec_triton.__all__ == ["TritonCodecStream", "is_available"]
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "is_available", lambda: False)
        assert codec_triton.is_available() is False
    with monkeypatch.context() as patch:
        patch.setattr(codec_triton, "find_spec", lambda name: None)
        assert codec_triton.is_available() is False


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        codec_triton.missing


@pytest.mark.skipif(find_spec("triton") is None, reason="needs triton")
def test_launch_deepest_steps_down_and_remembers(monkeypatch):
    """A depth whose pipeline does not fit raises OutOfResources at compile time; the next one is used."""
    from triton.runtime.errors import OutOfResources

    from haru.codec_triton import kernels

    monkeypatch.setattr(kernels, "_WINDOW_STAGES", {})
    tried = []

    def launch(stages):
        tried.append(stages)
        if stages > 2:
            raise OutOfResources(114_688, 101_376, "shared memory")

    assert kernels.launch_deepest(("4090",), launch) == 2 and tried == [3, 2]
    assert kernels.launch_deepest(("4090",), launch) == 2 and tried == [3, 2, 2]  # remembered
    assert kernels.launch_deepest(("h100",), lambda stages: None) == 3

    def never(stages):
        raise OutOfResources(1, 0, "shared memory")

    with pytest.raises(OutOfResources):
        kernels.launch_deepest(("tiny",), never)


def test_open_stream_rejects_unknown_backend():
    from haru.models.vibevoice_codec import open_stream

    with pytest.raises(ValueError):
        open_stream(None, backend="cuda")


@pytest.fixture(scope="module")
def codec():
    from haru import HaruModel

    return HaruModel.from_pretrained(os.environ["HARU_MODEL"], device="cuda").codec


def voice(frames: int, seed: int) -> torch.Tensor:
    """A deterministic speech-like signal: a gliding harmonic tone under a syllable-rate envelope, plus noise."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(frames * FRAME) / 24_000
    pitch = 120 + 60 * torch.sin(2 * torch.pi * 0.4 * t + seed)
    phase = 2 * torch.pi * torch.cumsum(pitch, 0) / 24_000
    tone = sum(torch.sin(k * phase) / k for k in range(1, 9))
    envelope = torch.sin(2 * torch.pi * 2.5 * t + seed).clamp(min=0) ** 2
    return 0.15 * envelope * tone + 0.01 * torch.randn(t.shape, generator=g)


def snr_db(reference: torch.Tensor, signal: torch.Tensor) -> float:
    reference, signal = reference.float(), signal.float()
    return float(10 * torch.log10(reference.pow(2).sum() / (reference - signal).pow(2).sum()))


def run_both(codec, audio: torch.Tensor, reset: tuple[int, list[int]] | None = None, num_stages: int | None = None):
    """Encode and decode ``audio [B, n * FRAME]`` one frame at a time on both backends (same latents decoded)."""
    from haru.models.vibevoice_codec import CodecStream

    reference = CodecStream(codec, audio.shape[0])
    stream = codec_triton.TritonCodecStream(codec, audio.shape[0], num_stages)
    outputs: list[list[torch.Tensor]] = [[], [], [], []]
    for i in range(audio.shape[1] // FRAME):
        if reset is not None and i == reset[0]:
            reference.reset(reset[1])
            stream.reset(reset[1])
        frame = audio[:, i * FRAME:(i + 1) * FRAME]
        means = reference.encode(frame)
        results = (means, stream.encode(frame), reference.decode(means), stream.decode(means))
        for collected, result in zip(outputs, results):
            collected.append(result)
    return [torch.cat(collected, dim=1) for collected in outputs]


@gpu
@pytest.mark.parametrize("batch", [1, 4])
def test_matches_torch_stream(codec, batch):
    """Rows start at different points of the signal; with several rows, one is reset mid-stream."""
    frames = 100
    audio = torch.stack([voice(frames, seed) for seed in range(batch)])
    reset = (40, [batch - 1]) if batch > 1 else None
    means, triton_means, wave, triton_wave = run_both(codec, audio, reset)
    assert triton_means.shape == means.shape == (batch, frames, 64) and triton_means.dtype == means.dtype
    assert triton_wave.shape == wave.shape == (batch, frames * FRAME) and triton_wave.dtype == wave.dtype
    assert snr_db(means, triton_means) > 30
    assert snr_db(wave, triton_wave) > 30


@gpu
@pytest.mark.parametrize("num_stages", [1, 2])
def test_explicit_pipeline_depth_matches_torch_stream(codec, num_stages):
    frames = 20
    means, triton_means, wave, triton_wave = run_both(codec, voice(frames, 0)[None], num_stages=num_stages)
    assert snr_db(means, triton_means) > 30
    assert snr_db(wave, triton_wave) > 30


@gpu
def test_invalid_pipeline_depth_raises(codec):
    with pytest.raises(ValueError):
        codec_triton.TritonCodecStream(codec, num_stages=4)


@gpu
def test_row_reset_equals_fresh_stream(codec):
    audio = torch.stack([voice(12, seed) for seed in range(3)])
    stream = codec_triton.TritonCodecStream(codec, 3)
    fresh = codec_triton.TritonCodecStream(codec, 3)
    stream.decode(stream.encode(audio[:, :6 * FRAME]))
    stream.reset([1])
    tail = audio[:, 6 * FRAME:]
    means, fresh_means = stream.encode(tail), fresh.encode(tail)
    assert torch.equal(means[1], fresh_means[1]) and not torch.equal(means[0], fresh_means[0])
    wave, fresh_wave = stream.decode(means), fresh.decode(means)
    assert torch.equal(wave[1], fresh_wave[1]) and not torch.equal(wave[0], fresh_wave[0])
    stream.reset()
    assert torch.equal(stream.encode(tail), fresh_means)


@gpu
def test_several_frames_per_call_equal_single_frames(codec):
    audio = voice(5, 0)[None]
    whole, single = codec_triton.TritonCodecStream(codec), codec_triton.TritonCodecStream(codec)
    means = whole.encode(audio)
    assert torch.equal(means, torch.cat([single.encode(audio[:, i * FRAME:(i + 1) * FRAME]) for i in range(5)], 1))
    assert torch.equal(whole.decode(means), torch.cat([single.decode(means[:, i:i + 1]) for i in range(5)], 1))
    with pytest.raises(ValueError):
        whole.encode(audio[:, :FRAME + 1])
    with pytest.raises(ValueError):
        whole.encode(torch.zeros(2, FRAME))


@gpu
def test_usable_from_another_thread(codec):
    audio = voice(2, 0)[None]
    stream = codec_triton.TritonCodecStream(codec)
    stream.warmup()
    expected = stream.encode(audio)
    stream.reset()
    results = []

    def work():
        results.append(stream.encode(audio))
        results.append(codec_triton.TritonCodecStream(codec).encode(audio))  # built in this thread

    thread = threading.Thread(target=work)
    thread.start()
    thread.join()
    assert len(results) == 2 and all(torch.equal(result, expected) for result in results)

