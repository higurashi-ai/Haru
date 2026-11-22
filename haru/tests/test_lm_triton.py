"""Triton language model backend: parity with the PyTorch ``LMStream`` on a real checkpoint
(``HARU_MODEL=/path/to/Haru-4B pytest -m gpu tests/test_lm_triton.py``)."""

import numpy as np
import pytest
import torch

from haru.models.haru import LMStream, open_lm_stream

FRAME = 3200


def test_open_lm_stream_rejects_unknown_backend():
    with pytest.raises(ValueError):
        open_lm_stream(None, backend="cuda")


@pytest.fixture(scope="module")
def prompt(model):
    from haru.prompt import build_prompt

    t = torch.arange(4 * 24_000) / 24_000
    voice = 0.2 * torch.sin(2 * torch.pi * 150 * t) * (1 + torch.sin(2 * torch.pi * 3 * t))
    generator = torch.Generator(device=model.device).manual_seed(0)
    return build_prompt(model.config, model.codec, model.tokenizer, "You are Haru.", voice, generator=generator)


def inputs(model, frames: int) -> list[tuple[torch.Tensor, ...]]:
    g = torch.Generator(device=model.device).manual_seed(1)
    dim, pad = model.config.latent_dim, torch.tensor([model.config.pad_token_id], device=model.device)
    return [
        (
            torch.randint(0, model.lm.config.vocab_size, (1,), device=model.device, generator=g) if i % 3 else pad,
            torch.randn(1, dim, device=model.device, generator=g),
            torch.randn(1, 1, dim, device=model.device, generator=g),
            torch.randn(1, dim, device=model.device, dtype=model.dtype, generator=g),
        )
        for i in range(frames)
    ]


def snr_db(reference: torch.Tensor, signal: torch.Tensor) -> float:
    reference, signal = reference.float(), signal.float()
    return float(10 * torch.log10(reference.pow(2).sum() / (reference - signal).pow(2).sum().clamp_min(1e-30)))


@pytest.mark.gpu("triton")
def test_matches_torch_stream(model, prompt):
    """Both streams see the same inputs; each keeps its own state, so rounding differences accumulate."""
    reference, stream = LMStream(model, 2.0), open_lm_stream(model, "triton", 2.0)
    assert type(stream) is not LMStream
    assert torch.equal(reference.prefill(prompt), stream.prefill(prompt))
    hidden_snr = []
    for token, latent, user_latent, noise in inputs(model, 30):
        expected = reference.step(token, latent, user_latent)
        hidden = stream.step(token, latent, user_latent)
        assert hidden.shape == expected.shape and hidden.dtype == expected.dtype
        hidden_snr.append(snr_db(expected, hidden))
        logits, expected_logits = stream.logits(expected), reference.logits(expected)
        assert logits.shape == expected_logits.shape and logits.dtype == torch.float32
        assert snr_db(expected_logits, logits) > 60
        sampled, expected_latent = stream.latent(expected, noise), reference.latent(expected, noise)
        assert sampled.shape == expected_latent.shape and sampled.dtype == expected_latent.dtype
        assert snr_db(expected_latent, sampled) > 30
    # A bfloat16 unit is 1-2 at the magnitude of the residual stream: single frames dip when a large value flips.
    assert np.median(hidden_snr) > 25


@pytest.mark.gpu("triton")
def test_restart_and_buffer_growth_do_not_change_the_output(model, prompt, monkeypatch):
    from haru import lm_triton

    frames = inputs(model, 10)
    stream = open_lm_stream(model, "triton")
    stream.prefill(prompt)
    first = [stream.step(*frame[:3]) for frame in frames]
    monkeypatch.setattr(lm_triton, "KV_CHUNK", 4)  # the key / value buffers now fill up twice
    stream.prefill(prompt)
    again = [stream.step(*frame[:3]) for frame in frames]
    assert stream.length == prompt.length + len(frames)
    assert all(torch.equal(a, b) for a, b in zip(first, again))


@pytest.mark.gpu("triton")
def test_session_on_triton_is_reproducible(model):
    from haru import DuplexConfig, DuplexSession

    def run() -> list:
        config = DuplexConfig(seed=7, speak_first=True, max_frames=8, lm_backend="triton", codec_backend="auto")
        session = DuplexSession(model, "You are Haru. Greet the other person.", config=config)
        frames = [session.start()]
        while not frames[-1].done:
            frames.append(session.step(np.zeros(FRAME, dtype=np.float32)))
        return frames

    first, second = run(), run()
    assert all(f.audio.shape == (FRAME,) and np.isfinite(f.audio).all() for f in first)
    assert [f.token for f in first] == [f.token for f in second]
    assert "".join(f.text for f in first).strip()
