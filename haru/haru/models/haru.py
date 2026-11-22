"""Haru: a full-duplex spoken dialogue model.

One language-model position is one 133 ms frame. Its input is the sum of three
streams at that frame: the model's text token, the model's audio latent and the
user's audio latent. The hidden state at frame ``t`` predicts frame ``t + 1``:
the text token through the tied output embedding and the audio latent through
the MeanFlow head.
"""

from __future__ import annotations

import inspect

import torch
from torch import nn
from transformers import DynamicCache, Qwen3_5TextConfig, Qwen3_5TextModel

from ..config import HaruConfig
from .meanflow_head import MeanFlowHead
from .vibevoice_codec import CODEC_BACKENDS, VibeVoiceCodec

_PREFILL_CHUNK = 512


class AudioProjector(nn.Module):
    """Audio latent -> LM embedding space."""

    def __init__(self, latent_dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden)
        self.norm = nn.RMSNorm(hidden, eps=1e-6)
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.norm(self.fc1(x)))


def _lm_config(params: dict) -> Qwen3_5TextConfig:
    accepted = inspect.signature(Qwen3_5TextConfig.__init__).parameters
    return Qwen3_5TextConfig(**{k: v for k, v in params.items() if k in accepted})


class HaruModel(nn.Module):
    def __init__(self, config: HaruConfig):
        super().__init__()
        self.config = config
        self.lm = Qwen3_5TextModel(_lm_config(config.lm))
        self.model_audio_proj = AudioProjector(config.latent_dim, config.hidden_size)
        self.user_audio_proj = AudioProjector(config.latent_dim, config.hidden_size)
        self.head = MeanFlowHead(config.head)
        self.codec = VibeVoiceCodec(config.codec)
        self.register_buffer("latent_scale", torch.ones(()))
        self.register_buffer("latent_bias", torch.zeros(()))
        self.tokenizer = None

    @classmethod
    def from_pretrained(cls, path_or_repo: str, device: str | torch.device | None = None, **kwargs) -> "HaruModel":
        """Load from a local directory or a Hugging Face repository."""
        from .loaders import load_model

        return load_model(path_or_repo, device=device, **kwargs)

    @property
    def device(self) -> torch.device:
        return self.latent_scale.device

    @property
    def dtype(self) -> torch.dtype:
        return self.model_audio_proj.fc1.weight.dtype

    # --------------------------------------------------------------- latents

    def scale_latents(self, raw: torch.Tensor) -> torch.Tensor:
        """Codec latents -> the normalized space of the head and projectors."""
        return (raw.float() + self.latent_bias.float()) * self.latent_scale.float()

    def unscale_latents(self, normalized: torch.Tensor) -> torch.Tensor:
        return normalized.float() / self.latent_scale.float() - self.latent_bias.float()

    # ------------------------------------------------------------ backbone

    def new_cache(self) -> DynamicCache:
        return DynamicCache(config=self.lm.config)

    def embed(
        self,
        tokens: torch.Tensor,
        model_latents: torch.Tensor | None = None,
        user_latents: torch.Tensor | None = None,
        model_mask: torch.Tensor | None = None,
        user_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sum the three streams: ``tokens [B, T]`` and raw latents ``[B, T, D]`` -> ``[B, T, H]``.

        A mask ``[B, T]`` zeroes an audio stream at positions that have none
        (the system prompt has neither, the voice reference has no user audio).
        """
        x = self.lm.embed_tokens(tokens)
        for proj, latents, mask in (
            (self.model_audio_proj, model_latents, model_mask),
            (self.user_audio_proj, user_latents, user_mask),
        ):
            if latents is None:
                continue
            y = proj(self.scale_latents(latents).to(x.dtype))
            if mask is not None:
                y = y * mask[..., None].to(y.dtype)
            x = x + y
        return x

    def forward(self, embeddings: torch.Tensor, cache: DynamicCache) -> torch.Tensor:
        """Run new positions ``[B, T, H]`` on top of ``cache`` -> hidden states ``[B, T, H]``."""
        return self.lm(inputs_embeds=embeddings, past_key_values=cache, use_cache=True).last_hidden_state

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Text logits ``[..., vocab]`` in float32."""
        return (hidden @ self.lm.embed_tokens.weight.T).float()

    def logits_fp32(self, hidden: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Logits of a few ``tokens`` computed entirely in float32.

        The PAD / EOP decision (stay silent or start speaking) is often closer
        than the resolution of bfloat16 at the magnitude of those logits.
        """
        return hidden.float() @ self.lm.embed_tokens.weight[tokens].float().T


class LMStream:
    """Language model state for one conversation."""

    def __init__(self, model: HaruModel, guidance_scale: float | None = None):
        self.model = model
        self.guidance_scale = guidance_scale
        self.cache: DynamicCache | None = None

    def prefill(self, prompt) -> torch.Tensor:
        """Reset the cache and return the prompt's last hidden state ``[1, H]``."""
        model = self.model
        self.cache = model.new_cache()
        hidden = None
        for s in range(0, prompt.length, _PREFILL_CHUNK):
            span = slice(s, s + _PREFILL_CHUNK)
            embeddings = model.embed(
                prompt.tokens[None, span], prompt.model_latents[None, span], prompt.user_latents[None, span],
                prompt.model_mask[None, span], prompt.user_mask[None, span],
            )
            hidden = model(embeddings, self.cache)[:, -1]
        return hidden

    def step(self, token: torch.Tensor, latent: torch.Tensor, user_latent: torch.Tensor) -> torch.Tensor:
        """One frame: text token ``[1]``, raw model latent ``[1, D]`` and user latent ``[1, 1, D]`` -> ``[1, H]``."""
        return self.model(self.model.embed(token[None], latent[None], user_latent), self.cache)[:, -1]

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.logits(hidden)

    def latent(self, hidden: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Hidden state ``[1, H]`` and noise ``[1, D]`` -> raw audio latent ``[1, D]``."""
        return self.model.unscale_latents(self.model.head.sample(hidden, noise, self.guidance_scale))


def open_lm_stream(model: HaruModel, backend: str = "torch", guidance_scale: float | None = None) -> LMStream:
    """In auto mode, use Triton when available for a bfloat16 model on CUDA."""
    from .. import codec_triton

    if backend not in CODEC_BACKENDS:
        raise ValueError(f"unknown language model backend {backend!r}")
    usable = codec_triton.is_available() and model.device.type == "cuda" and model.dtype == torch.bfloat16
    if backend == "triton" and not usable:
        raise RuntimeError("the Triton language model backend needs triton and a bfloat16 model on CUDA")
    if backend != "torch" and usable:
        from ..lm_triton import TritonLMStream

        return TritonLMStream(model, guidance_scale)
    return LMStream(model, guidance_scale)
