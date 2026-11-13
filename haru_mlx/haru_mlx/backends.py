"""Engines that run conversations for the server.

A :class:`Backend` opens conversations; a :class:`Conversation` is one live
dialogue, stepped one 133 ms frame at a time. The server only talks to these
two protocols.

    backend = LocalBackend("higurashi-ai/Haru-4B")
    await backend.load()
    conversation = await backend.open(SessionSettings(system_prompt="..."))
    frame = await conversation.start()            # model frame 0
    frame = await conversation.step(user_pcm)     # user frame k in, model frame k + 1 out
    await conversation.close()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .frames import DuplexConfig, Frame
from .hub import DEFAULT_REPO
from .worker import MLXWorker

if TYPE_CHECKING:
    from .session import DuplexSession

logger = logging.getLogger(__name__)

_POOL_BYTES = 1 << 30


@dataclass
class SessionSettings:
    """What one conversation is started with."""

    system_prompt: str = ""
    reference: np.ndarray | None = None  # voice of the model: mono float32 at 24 kHz
    guidance_scale: float = 1.5
    temperature: float = 0.7
    top_k: int = 25
    top_p: float = 1.0
    speak_first: bool = False
    max_frames: int = 27_000  # 60 minutes
    seed: int | None = None


class Conversation(Protocol):
    """One live conversation. Calls are sequential: ``start`` once, then ``step`` per user frame."""

    async def start(self) -> Frame:
        """Run the prompt and return model frame 0."""
        ...

    async def step(self, user_audio: np.ndarray) -> Frame:
        """Consume one user frame (3200 float32 samples at 24 kHz); return the next model frame."""
        ...

    async def close(self) -> None:
        """Release the conversation's resources."""
        ...


class Backend(Protocol):
    async def load(self) -> None:
        """Load and warm up the engine; ``ready()`` is true afterwards."""
        ...

    def ready(self) -> bool: ...

    async def open(self, settings: SessionSettings) -> Conversation: ...


class LocalConversation:
    """A :class:`DuplexSession` whose calls run on the backend's MLX thread."""

    def __init__(self, backend: "LocalBackend", session: "DuplexSession"):
        self._backend = backend
        self._session: DuplexSession | None = session

    async def start(self) -> Frame:
        return await self._backend.worker.run(self._session.start)

    async def step(self, user_audio: np.ndarray) -> Frame:
        arrived = time.perf_counter()
        frame = await self._backend.worker.run(self._session.step, user_audio)
        # The next user frame completes one frame after this one did.
        self._backend.worker.expect(arrived + self._backend.frame_seconds)
        return frame

    async def close(self) -> None:
        self._backend.worker.expect(None)
        self._session = None  # drops the conversation's cache


class LocalBackend:
    """Runs the MLX model in this process.

    All model work happens on one dedicated thread (see :class:`MLXWorker`), so
    the event loop keeps serving sockets while a frame is computed, and
    conversations that share the model take turns frame by frame.
    """

    def __init__(
        self,
        model: str = DEFAULT_REPO,
        quantize: int | None = None,
        warmup_frames: int = 16,
        warm_seconds: float = 0.03,
    ):
        """``model``: a checkpoint directory or a Hugging Face repository id.

        ``quantize``: 8 for an 8-bit backbone, or None for the checkpoint's
        precision. ``warm_seconds``: how long before a frame is
        due the MLX thread starts polling for it (0: never).
        """
        self.model_path = model
        self.quantize = quantize
        self.warmup_frames = warmup_frames
        self.model = None
        self.frame_seconds = 0.0
        self.worker = MLXWorker(warm=warm_seconds)
        self._ready = False

    async def load(self) -> None:
        await self.worker.run(self._load)
        self._ready = True

    def load_blocking(self) -> None:
        """``load()`` for callers without an event loop."""
        self.worker.submit(self._load).result()
        self._ready = True

    def ready(self) -> bool:
        return self._ready

    async def open(self, settings: SessionSettings) -> LocalConversation:
        if self.model is None:
            raise RuntimeError("the model is not loaded")
        return LocalConversation(self, self.session(settings))

    def session(self, settings: SessionSettings) -> "DuplexSession":
        """A new conversation; it must be driven on the MLX thread (``worker``)."""
        from .session import DuplexSession

        config = DuplexConfig(
            guidance_scale=settings.guidance_scale,
            temperature=settings.temperature,
            top_k=settings.top_k,
            top_p=settings.top_p,
            speak_first=settings.speak_first,
            max_frames=settings.max_frames,
            seed=settings.seed,
        )
        return DuplexSession(self.model, settings.system_prompt, settings.reference, config)

    def _load(self) -> None:
        import mlx.core as mx

        from .models import load_model

        started = time.perf_counter()
        if mx.metal.is_available():
            # Keep the weights resident: a frame reads all of them.
            mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
        # Frames reuse small buffers. Without a limit, each regrowth of the
        # attention cache would leave its old buffers pooled for the session.
        mx.set_cache_limit(_POOL_BYTES)
        self.model = load_model(self.model_path, quantize=self.quantize)
        self.frame_seconds = 1.0 / self.model.config.frame_rate
        precision = f"{self.quantize}-bit backbone" if self.quantize else str(self.model.dtype)
        logger.info("loaded %s (%s) in %.1f s", self.model_path, precision, time.perf_counter() - started)
        if self.warmup_frames > 0:
            self._warm_up()

    def _warm_up(self) -> None:
        """A throwaway conversation: the first call of every code path and shape is slow.

        Its prompt is as long as a typical one (a card of a few hundred tokens
        and a 30 s voice), so the first real conversation starts without delay.
        """
        cfg = self.model.config
        rng = np.random.default_rng(0)
        reference = (0.01 * rng.standard_normal(30 * cfg.sample_rate)).astype(np.float32)
        card = "This conversation only warms up the model. " * 40
        session = self.session(SessionSettings(system_prompt=card, reference=reference, speak_first=True))
        prompt_seconds = session.start().seconds
        silence = np.zeros(cfg.frame_samples, dtype=np.float32)
        times = [session.step(silence).seconds for _ in range(self.warmup_frames)]
        logger.info(
            "warm-up: prompt of %d positions in %.1f s, then %.1f ms per frame (a frame is %.0f ms of audio)",
            session.prompt_length, prompt_seconds, 1000 * float(np.median(times)), 1000 / cfg.frame_rate,
        )
