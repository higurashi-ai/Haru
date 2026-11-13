"""A live full-duplex conversation: one user frame in, one model frame out.

    session = DuplexSession(model, system_prompt="...", reference=voice_clip)
    frame = session.start()              # model frame 0
    while True:
        frame = session.step(user_pcm)   # 3200 samples in, 3200 samples out

Frame ``t`` of the model comes from the hidden state at the previous position:
its text token from the output embedding and its audio latent from the MeanFlow
head, decoded to audio right away. When user frame ``t`` arrives, the position
``text[t] + model_audio[t] + user_audio[t]`` runs and yields frame ``t + 1``.
Frame 0 comes from the end of the prompt, so the model stays exactly one frame
ahead and the frame length (133 ms) is the only structural latency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .models.haru import HaruModel
from .models.vibevoice_codec import open_stream
from .prompt import build_prompt
from .sampling import DryConfig, DryState, StreamDetokenizer, sample_token

_PREFILL_CHUNK = 512


@dataclass
class DuplexConfig:
    guidance_scale: float = 1.5  # audio guidance (valid range 1-2)
    temperature: float = 0.7
    top_k: int = 25
    top_p: float = 1.0
    dry: DryConfig | None = field(default_factory=DryConfig)
    eop_bias: float = 0.0  # added to the EOP logit: > 0 makes the model start speaking sooner
    speak_first: bool = False  # the model opens the conversation
    max_frames: int = 27_000  # 60 minutes
    reference_seconds: float = 180.0
    seed: int | None = None
    codec_backend: str = "torch"  # "torch", or "triton" (faster; needs the triton package)


@dataclass
class Frame:
    audio: np.ndarray  # float32 [3200] at 24 kHz
    token: int
    text: str  # new text of this frame, "" if none
    index: int
    seconds: float  # compute time
    done: bool = False


class DuplexSession:
    def __init__(
        self,
        model: HaruModel,
        system_prompt: str = "",
        reference: np.ndarray | torch.Tensor | None = None,
        config: DuplexConfig | None = None,
    ):
        """``reference``: mono 24 kHz clip of the voice the model should speak with."""
        self.model = model
        self.config = config or DuplexConfig()
        self.system_prompt = system_prompt
        cfg = model.config
        self.frame_samples = cfg.frame_samples
        self.pad_id, self.eop_id = cfg.pad_token_id, cfg.eop_token_id
        self._silent = {self.pad_id, self.eop_id}
        self._stream_ids = torch.tensor([self.pad_id, self.eop_id], device=model.device)
        self._banned = sorted(self._silent | ({cfg.eos_token_id} if cfg.eos_token_id is not None else set()))

        self.reference = None
        if reference is not None and len(reference):
            limit = int(self.config.reference_seconds * cfg.sample_rate)
            self.reference = torch.as_tensor(reference, dtype=torch.float32).reshape(-1)[-limit:]

        self.index = 0
        self.prompt_length = 0
        self._generator = torch.Generator(device=model.device)
        self._cache = None
        self._codec = open_stream(model.codec, backend=self.config.codec_backend)
        self._token: torch.Tensor | None = None
        self._latent: torch.Tensor | None = None
        self._force_text = 0
        self._dry = DryState(self.config.dry, set(self._banned)) if self.config.dry else None
        self._detokenizer = StreamDetokenizer(model.tokenizer, self._silent)

    # ------------------------------------------------------------- prompt

    @torch.no_grad()
    def start(self) -> Frame:
        """Run the prompt (system prompt + voice reference) and produce frame 0."""
        started = time.perf_counter()
        model, cfg = self.model, self.config
        if cfg.seed is None:
            self._generator.seed()
        else:
            self._generator.manual_seed(cfg.seed)
        self._codec.reset()
        self._cache = model.new_cache()

        prompt = build_prompt(
            model.config, model.codec, model.tokenizer, self.system_prompt, self.reference, cfg.speak_first, self._generator
        )
        self._force_text = 1 if cfg.speak_first else 0
        n = prompt.length

        hidden = None
        for s in range(0, n, _PREFILL_CHUNK):
            e = min(n, s + _PREFILL_CHUNK)
            span = slice(s, e)
            embeddings = model.embed(
                prompt.tokens[None, span], prompt.model_latents[None, span], prompt.user_latents[None, span],
                prompt.model_mask[None, span], prompt.user_mask[None, span],
            )
            hidden = model(embeddings, self._cache)[:, -1]
        self.prompt_length = n
        self.index = 0
        return self._emit(hidden, started)

    # ------------------------------------------------------------- frames

    @torch.no_grad()
    def step(self, user_audio: np.ndarray | torch.Tensor) -> Frame:
        """Consume one user frame (3200 samples, float32, 24 kHz); return the next model frame."""
        if self._latent is None:
            raise RuntimeError("call start() first")
        started = time.perf_counter()
        model = self.model
        audio = torch.as_tensor(user_audio, dtype=torch.float32).reshape(1, -1)
        if audio.shape[1] != self.frame_samples:
            raise ValueError(f"a user frame is {self.frame_samples} samples, got {audio.shape[1]}")
        user_latent = model.codec.sample(self._codec.encode(audio), self._generator).float()
        embeddings = model.embed(self._token[None], self._latent[None], user_latent)
        hidden = model(embeddings, self._cache)[:, -1]
        return self._emit(hidden, started)

    def _emit(self, hidden: torch.Tensor, started: float) -> Frame:
        """Sample the next frame's text token and audio from ``hidden [1, H]``."""
        model, cfg = self.model, self.config
        logits = model.logits(hidden)
        if self._dry is not None:
            self._dry.apply(logits[0])
        logits[:, self._stream_ids] = model.logits_fp32(hidden, self._stream_ids)
        if cfg.eop_bias:
            logits[:, self.eop_id] += cfg.eop_bias
        if self._force_text > 0:
            logits[:, self._banned] = float("-inf")
            self._force_text -= 1
        token = sample_token(logits, cfg.temperature, cfg.top_k, cfg.top_p, self._generator)

        noise = torch.randn(
            1, model.config.latent_dim, device=hidden.device, dtype=model.dtype, generator=self._generator
        )
        latent = model.unscale_latents(model.head.sample(hidden, noise, cfg.guidance_scale))
        audio = self._codec.decode(latent[None])[0].float().clamp(-1.0, 1.0)

        self._token, self._latent = token, latent
        token_id = int(token)
        if self._dry is not None:
            self._dry.update(token_id)
        frame = Frame(
            audio=audio.cpu().numpy(),
            token=token_id,
            text=self._detokenizer.push(token_id),
            index=self.index,
            seconds=time.perf_counter() - started,
        )
        self.index += 1
        frame.done = self.index >= cfg.max_frames
        return frame
