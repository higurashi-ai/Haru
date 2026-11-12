"""Engines that run conversations for the server.

A :class:`Backend` opens conversations; a :class:`Conversation` is one live
dialogue, stepped one 133 ms frame at a time. The server only talks to these
two protocols, so the same web server can drive the in-process PyTorch model
(:class:`LocalBackend`) or a vLLM-Omni server (:class:`VllmBackend`).

    backend = LocalBackend("higurashi-ai/Haru-4B")
    await backend.load()
    conversation = await backend.open(SessionSettings(system_prompt="..."))
    frame = await conversation.start()            # model frame 0
    frame = await conversation.step(user_pcm)     # user frame k in, model frame k + 1 out
    await conversation.close()
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

import aiohttp
import numpy as np

from .audio import SAMPLE_RATE
from .models.loaders import DEFAULT_REPO
from .session import DuplexConfig, DuplexSession, Frame

logger = logging.getLogger(__name__)

FRAME_SAMPLES = 3200


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
    """A :class:`DuplexSession` whose calls run on the backend's worker thread."""

    def __init__(self, backend: "LocalBackend", session: DuplexSession):
        self._backend = backend
        self._session: DuplexSession | None = session

    async def start(self) -> Frame:
        return await self._backend.run(self._session.start)

    async def step(self, user_audio: np.ndarray) -> Frame:
        return await self._backend.run(self._session.step, user_audio)

    async def close(self) -> None:
        self._session = None  # drops the conversation's cache


class LocalBackend:
    """Runs the PyTorch model in this process.

    All model work happens on one dedicated worker thread, so the event loop
    keeps serving sockets while a frame is computed, and conversations that
    share the model take turns frame by frame.
    """

    def __init__(self, model: str = DEFAULT_REPO, device: str | None = None, warmup_frames: int = 16):
        """``model``: a checkpoint directory or a Hugging Face repository id."""
        self.model_path = model
        self.device = device
        self.warmup_frames = warmup_frames
        self.model = None
        self._ready = False
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="haru-model")

    async def run(self, fn, *args):
        """Call ``fn(*args)`` on the worker thread."""
        return await asyncio.get_running_loop().run_in_executor(self._worker, fn, *args)

    async def load(self) -> None:
        await self.run(self._load)
        self._ready = True

    def ready(self) -> bool:
        return self._ready

    async def open(self, settings: SessionSettings) -> LocalConversation:
        if self.model is None:
            raise RuntimeError("the model is not loaded")
        return LocalConversation(self, self._session(settings))

    def _session(self, settings: SessionSettings) -> DuplexSession:
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
        from .models import load_model

        started = time.perf_counter()
        self.model = load_model(self.model_path, device=self.device)
        logger.info("loaded %s on %s in %.1f s", self.model_path, self.model.device, time.perf_counter() - started)
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
        session = self._session(SessionSettings(system_prompt=card, reference=reference, speak_first=True))
        prompt_seconds = session.start().seconds
        silence = np.zeros(cfg.frame_samples, dtype=np.float32)
        times = [session.step(silence).seconds for _ in range(self.warmup_frames)]
        logger.info(
            "warm-up: prompt of %d positions in %.1f s, then %.1f ms per frame (a frame is %.0f ms of audio)",
            session.prompt_length, prompt_seconds, 1000 * float(np.median(times)), 1000 / cfg.frame_rate,
        )


class VllmConversation:
    """One realtime session on a Haru vLLM-Omni server (``/v1/realtime?duplex=1``)."""

    def __init__(self, http: aiohttp.ClientSession, url: str, settings: SessionSettings):
        self._http = http
        self._url = url
        self._settings = settings
        self._socket: aiohttp.ClientWebSocketResponse | None = None
        self._reader: asyncio.Task | None = None
        self._audio: asyncio.Queue[np.ndarray | Exception] = asyncio.Queue()
        self._text = ""
        self._index = 0

    async def start(self) -> Frame:
        settings = self._settings
        session = {
            "modalities": ["audio", "text"],
            "instructions": settings.system_prompt,
            "input_audio_format": "pcm_f32le",
            "output_audio_format": "pcm16",
            "sample_rate_hz": SAMPLE_RATE,
            "audio": {"input": {"sample_rate_hz": SAMPLE_RATE}, "output": {"sample_rate_hz": SAMPLE_RATE}},
            "turn_detection": None,
            "temperature": settings.temperature,
            "extra_body": {
                "auto_response": True,
                "guidance_scale": settings.guidance_scale,
                "top_k": settings.top_k,
                "top_p": settings.top_p,
                "speak_first": settings.speak_first,
                "seed": settings.seed,
            },
        }
        if settings.reference is not None and len(settings.reference):
            session["ref_audio"] = _wav_data_uri(settings.reference)
        self._socket = await self._http.ws_connect(self._url, max_msg_size=0, heartbeat=20)
        await self._socket.send_json({"type": "session.update", "session": session})
        while True:
            event = await self._socket.receive_json()
            if event.get("type") == "session.created":
                break
            if event.get("type") == "error":
                raise RuntimeError(_error_message(event))
        self._reader = asyncio.create_task(self._read())
        # The first append starts the session: the prompt runs and frame 0 comes back.
        return await self.step(np.zeros(FRAME_SAMPLES, dtype=np.float32))

    async def step(self, user_audio: np.ndarray) -> Frame:
        started = time.perf_counter()
        pcm = np.asarray(user_audio, dtype="<f4").reshape(-1)
        await self._socket.send_json({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm.tobytes()).decode("ascii"),
            "format": "pcm_f32le",
            "sample_rate_hz": SAMPLE_RATE,
        })
        audio = await self._audio.get()
        if isinstance(audio, Exception):
            raise audio
        text, self._text = self._text, ""
        frame = Frame(audio=audio, token=-1, text=text, index=self._index, seconds=time.perf_counter() - started)
        self._index += 1
        frame.done = self._index >= self._settings.max_frames
        return frame

    async def _read(self) -> None:
        try:
            async for message in self._socket:
                if message.type != aiohttp.WSMsgType.TEXT:
                    continue
                event = json.loads(message.data)
                kind = event.get("type")
                if kind == "response.output_audio.delta":
                    pcm = np.frombuffer(base64.b64decode(event["delta"]), dtype="<i2")
                    self._audio.put_nowait(pcm.astype(np.float32) / 32768.0)
                elif kind == "response.output_audio_transcript.delta":
                    self._text += event.get("delta", "")
                elif kind == "error":
                    self._audio.put_nowait(RuntimeError(_error_message(event)))
            self._audio.put_nowait(ConnectionError("the vLLM-Omni server closed the session"))
        except Exception as error:  # surfaced to the caller of step()
            self._audio.put_nowait(error)

    async def close(self) -> None:
        if self._socket is not None and not self._socket.closed:
            try:
                await self._socket.send_json({"type": "session.close"})
            except ConnectionError:
                pass
            await self._socket.close()
        if self._reader is not None:
            self._reader.cancel()


class VllmBackend:
    """Runs conversations on a Haru vLLM-Omni server (see the ``haru_vllm`` package).

    ``url`` is the server's base URL, e.g. ``http://localhost:8000``.
    """

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self._http: aiohttp.ClientSession | None = None
        self._ready = False

    async def load(self) -> None:
        self._http = aiohttp.ClientSession()
        while True:
            try:
                async with self._http.get(f"{self.url}/health") as response:
                    if response.status == 200:
                        break
            except aiohttp.ClientError:
                pass
            logger.info("waiting for the vLLM-Omni server at %s", self.url)
            await asyncio.sleep(5)
        self._ready = True

    def ready(self) -> bool:
        return self._ready

    async def open(self, settings: SessionSettings) -> VllmConversation:
        if self._http is None:
            raise RuntimeError("the backend is not loaded")
        realtime = self.url.replace("http", "ws", 1) + "/v1/realtime?duplex=1"
        return VllmConversation(self._http, realtime, settings)


def _wav_data_uri(audio: np.ndarray) -> str:
    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, np.asarray(audio, dtype=np.float32), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _error_message(event: dict) -> str:
    error = event.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or error)
    return str(error or event)
