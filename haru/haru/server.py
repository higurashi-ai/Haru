"""Web demo: talk to Haru from a browser.

    haru-server --hf-repo higurashi-ai/Haru-4B        # then open http://localhost:8998

Serves the web client and its API. A conversation is registered with
``POST /api/session`` (system prompt, voice, settings) and then runs over one
of two transports:

* ``GET /api/chat?session_id=...`` — a WebSocket of binary messages, one kind
  byte followed by the payload: ``0x00`` handshake (server, once the
  conversation can start), ``0x01`` Ogg/Opus pages (24 kHz mono, both
  directions), ``0x02`` UTF-8 text of the model, ``0x06`` ping.
* ``POST /api/webrtc/offer?session_id=...`` — WebRTC (see :mod:`haru.webrtc`).

The model stays one frame ahead of the user: frame 0 is sent with the
handshake, and frame ``k + 1`` as soon as user frame ``k`` (133 ms) has
arrived. Conversations beyond ``--max-sessions`` wait for a free slot.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import secrets
import mimetypes
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import sphn
from aiohttp import web

from .audio import SAMPLE_RATE, decode_audio, prepare_reference, trim_silence
from .backends import Backend, Conversation, LocalBackend, SessionSettings, VllmBackend
from .models.loaders import DEFAULT_REPO
from .playout import PlayoutOptions, UplinkOptions
from .prompt import card_templates
from .session import Frame

try:
    from . import webrtc
except ImportError:  # the optional "webrtc" extra is not installed
    webrtc = None

logger = logging.getLogger("haru.server")

FRAME_SAMPLES = 3200
FRAME_MS = 1000.0 * FRAME_SAMPLES / SAMPLE_RATE
KIND_HANDSHAKE, KIND_AUDIO, KIND_TEXT, KIND_PING = 0x00, 0x01, 0x02, 0x06
PING_SECONDS = 3.0  # the client drops a connection that is silent for 10 s
SYSTEM_PROMPT_MAX_CHARS = 4000
DEFAULT_GUIDANCE_SCALE = 1.5
REFERENCE_MIN_SECONDS = 3.0
REFERENCE_MAX_SECONDS = 180.0
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_STORED = 32  # uploaded voices and registered sessions kept in memory
CLIENT_SILENCE_SECONDS = 15  # WebSocket: close when the microphone stream stops for this long
RTC_CONNECT_SECONDS = 45.0  # a peer that never connects must not keep its slot

# Buffering for the path between browser and server. "client" is sent to the
# browser (WebSocket playout queue, microphone page size); "playout" and
# "uplink" set the WebRTC buffers for model audio and microphone audio.
NETWORK_PROFILES: dict[str, dict[str, Any]] = {
    "internet": {
        "client": {"playout_start_ms": 2 * FRAME_MS, "catchup_target_ms": FRAME_MS + 100, "mic_frames_per_page": 2},
        "playout": {"start_margin_ms": 40.0},
        "uplink": {"playout_ms": 40.0},
    },
    "local": {
        "client": {"playout_start_ms": FRAME_MS, "catchup_target_ms": 60.0, "mic_frames_per_page": 1},
        "playout": {"start_margin_ms": 15.0},
        "uplink": {"playout_ms": 20.0},
    },
}
# WebRTC buffer settings a command-line flag can override: flag -> (group, field).
BUFFER_FLAGS = {
    "start_margin_ms": ("playout", "start_margin_ms"),
    "max_buffer_ms": ("playout", "max_buffer_ms"),
    "uplink_playout_ms": ("uplink", "playout_ms"),
    "uplink_playout_max_ms": ("uplink", "playout_max_ms"),
    "uplink_shrink_after_s": ("uplink", "shrink_after_s"),
}


def encode_message(kind: int, payload: bytes = b"") -> bytes:
    return bytes([kind]) + payload


def decode_message(data: bytes) -> tuple[int, bytes]:
    """``(kind, payload)`` of one WebSocket message; kind -1 when empty."""
    return (data[0], bytes(data[1:])) if data else (-1, b"")


class FrameBuffer:
    """Collects decoded user audio and hands it out in whole model frames."""

    def __init__(self, frame_samples: int = FRAME_SAMPLES):
        self.frame_samples = frame_samples
        self._pending = np.zeros(0, dtype=np.float32)

    def push(self, pcm: np.ndarray) -> list[np.ndarray]:
        self._pending = np.concatenate([self._pending, np.asarray(pcm, dtype=np.float32).reshape(-1)])
        count = len(self._pending) // self.frame_samples
        frames = [self._pending[i * self.frame_samples : (i + 1) * self.frame_samples] for i in range(count)]
        self._pending = self._pending[count * self.frame_samples :]
        return frames


@dataclass
class ServerOptions:
    static_dir: Path | None = None  # built web client; default: the bundled one
    max_sessions: int = 1
    time_limit_seconds: int = 3600
    default_language: str = "en"
    temperature: float = 0.7
    top_k: int = 25
    seed: int | None = None
    network_profile: str = "internet"
    buffers: dict[str, float] = field(default_factory=dict)  # BUFFER_FLAGS overrides of the profile
    webrtc: str = "auto"  # auto: when a TURN relay is configured; on; off
    ice_servers_file: str | None = None


class DemoAssets:
    """Card templates and example cards of the package, and default voices if any are installed.

    No voice ships with Haru: users upload or record one. A file
    ``assets/voices/<language>.flac`` is offered as that language's default voice.
    """

    def __init__(self) -> None:
        root = resources.files("haru") / "assets"
        templates = card_templates()
        self.languages: list[dict[str, str]] = templates["languages"]
        self.templates: dict[str, Any] = templates["templates"]
        self.presets: dict[str, list[dict[str, Any]]] = {}
        self.voice_files: dict[str, Any] = {}
        self.voices: dict[str, np.ndarray] = {}
        for code in (entry["code"] for entry in self.languages):
            cards = json.loads((root / "cards" / f"{code}.json").read_text(encoding="utf-8"))
            self.presets[code] = [{"id": key, "card": card} for key, card in cards.items()]
            voice = root / "voices" / f"{code}.flac"
            if voice.is_file():
                self.voice_files[code] = voice
                self.voices[code] = prepare_reference(decode_audio(voice.read_bytes()), REFERENCE_MAX_SECONDS)


def _error(status: int, code: str, message: str, **params: Any) -> web.Response:
    """An error for the setup page: it translates ``code`` (with ``params``) and falls back to ``error``."""
    return web.json_response({"code": code, "error": message, **params}, status=status)


class DemoServer:
    def __init__(self, backend: Backend, options: ServerOptions | None = None):
        self.backend = backend
        self.options = options or ServerOptions()
        if self.options.network_profile not in NETWORK_PROFILES:
            raise ValueError(f"network profile must be one of {sorted(NETWORK_PROFILES)}")
        self.profile = NETWORK_PROFILES[self.options.network_profile]
        self.assets = DemoAssets()
        if self.options.default_language not in self.assets.presets:
            raise ValueError(f"language must be one of {sorted(self.assets.presets)}")
        self.static_dir = self.options.static_dir or Path(str(resources.files("haru") / "static"))
        self.max_frames = max(1, self.options.time_limit_seconds * SAMPLE_RATE // FRAME_SAMPLES)
        self.references: OrderedDict[str, np.ndarray] = OrderedDict()
        self.sessions: OrderedDict[str, SessionSettings] = OrderedDict()
        self.slots = asyncio.Semaphore(self.options.max_sessions)
        # A conversation whose prompt already ran: (session id, conversation, frame 0).
        self._prepared: tuple[str, Conversation, Frame] | None = None
        self.rtc_sessions: dict[str, Any] = {}
        self.ice = None
        if webrtc is not None and self.options.webrtc != "off":
            self.ice = webrtc.IceServers(self.options.ice_servers_file, conversation_seconds=self.options.time_limit_seconds)
        self.http: aiohttp.ClientSession | None = None

    # ------------------------------------------------------------ lifecycle

    def build_app(self) -> web.Application:
        app = web.Application(client_max_size=MAX_UPLOAD_BYTES)
        app.cleanup_ctx.append(self._lifespan)
        app.router.add_get("/", self.handle_index)
        app.router.add_get("/assets/{name:.+}", self.handle_asset)
        app.router.add_get("/api/ready", self.handle_ready)
        app.router.add_get("/api/config", self.handle_config)
        app.router.add_get("/api/default_reference", self.handle_default_reference)
        app.router.add_post("/api/reference_audio", self.handle_reference_upload)
        app.router.add_post("/api/session", self.handle_create_session)
        app.router.add_get("/api/chat", self.handle_chat)
        app.router.add_get("/api/webrtc/config", self.handle_webrtc_config)
        app.router.add_post("/api/webrtc/offer", self.handle_webrtc_offer)
        app.router.add_post("/api/webrtc/cancel", self.handle_webrtc_cancel)
        app.router.add_post("/api/client_stats", self.handle_client_stats)
        return app

    async def _lifespan(self, _app: web.Application):
        self.http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
        loading = asyncio.create_task(self._load())
        yield
        loading.cancel()
        for session in list(self.rtc_sessions.values()):
            await session.close()
        await self._discard_prepared()
        await self.http.close()

    async def _load(self) -> None:
        try:
            await self.backend.load()
            logger.info("ready")
        except Exception:
            logger.exception("the model could not be loaded")

    # ------------------------------------------------------- static client

    def _file(self, path: Path) -> web.StreamResponse:
        if not path.is_file():
            raise web.HTTPNotFound(text="not found (is the web client built? see client/README.md)")
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return web.FileResponse(path, headers={"Content-Type": content_type})

    async def handle_index(self, _request: web.Request) -> web.StreamResponse:
        return self._file(self.static_dir / "index.html")

    async def handle_asset(self, request: web.Request) -> web.StreamResponse:
        name = request.match_info["name"]
        if ".." in Path(name).parts:
            raise web.HTTPNotFound()
        return self._file(self.static_dir / "assets" / name)

    # ----------------------------------------------------------------- API

    async def handle_ready(self, _request: web.Request) -> web.Response:
        return web.json_response({"ready": self.backend.ready()}, headers={"Cache-Control": "no-store"})

    def _default_reference_id(self, language: str) -> str:
        return "default" if language == self.options.default_language else f"default_{language}"

    def _webrtc_enabled(self) -> bool:
        if self.ice is None:
            return False
        return self.options.webrtc == "on" or self.ice.has_turn

    async def handle_config(self, _request: web.Request) -> web.Response:
        """Everything the setup page shows before a conversation."""
        assets, options = self.assets, self.options
        config = {
            "languages": assets.languages,
            "default_language": options.default_language,
            "card_templates": assets.templates,
            "card_presets": assets.presets,
            "system_prompt_max_chars": SYSTEM_PROMPT_MAX_CHARS,
            "reference_min_seconds": REFERENCE_MIN_SECONDS,
            "reference_max_seconds": REFERENCE_MAX_SECONDS,
            "time_limit_seconds": options.time_limit_seconds,
            "webrtc": self._webrtc_enabled(),
            "playout": self.profile["client"],
            "default_references": {
                code: {
                    "id": self._default_reference_id(code),
                    "seconds": round(len(voice) / SAMPLE_RATE, 2),
                    "trimmed": False,
                    "preview_url": f"/api/default_reference?lang={code}",
                }
                for code, voice in assets.voices.items()
            },
        }
        return web.json_response(config, headers={"Cache-Control": "no-store"})

    async def handle_default_reference(self, request: web.Request) -> web.Response:
        voice = self.assets.voice_files.get(request.query.get("lang", self.options.default_language))
        if voice is None:
            raise web.HTTPNotFound()
        return web.Response(body=voice.read_bytes(), content_type="audio/flac")

    def _reference(self, reference_id: str) -> np.ndarray | None:
        for code, voice in self.assets.voices.items():
            if reference_id == self._default_reference_id(code):
                return voice
        return self.references.get(reference_id)

    async def handle_reference_upload(self, request: web.Request) -> web.Response:
        """Store an uploaded or recorded voice (any audio file as the body); returns its id."""
        body = await request.read()
        if not body:
            return _error(400, "empty_upload", "The upload is empty.")
        try:
            audio = await asyncio.to_thread(decode_audio, body)
        except Exception:
            return _error(400, "unreadable_audio", "The audio file could not be read.")
        audio = trim_silence(audio)
        seconds = len(audio) / SAMPLE_RATE
        if seconds < REFERENCE_MIN_SECONDS:
            return _error(
                400, "audio_too_short",
                f"The audio has only {seconds:.1f} s of speech; at least {REFERENCE_MIN_SECONDS:g} s is needed.",
                seconds=round(seconds, 1), min_seconds=REFERENCE_MIN_SECONDS,
            )
        limit = int(REFERENCE_MAX_SECONDS * SAMPLE_RATE)
        trimmed = len(audio) > limit
        audio = audio[-limit:]  # the model was trained on the end of a speaker's audio
        reference_id = uuid.uuid4().hex
        self.references[reference_id] = audio
        while len(self.references) > MAX_STORED:
            self.references.popitem(last=False)
        return web.json_response({"id": reference_id, "seconds": round(len(audio) / SAMPLE_RATE, 2), "trimmed": trimmed})

    async def handle_create_session(self, request: web.Request) -> web.Response:
        """Register a conversation: ``{"system_prompt", "reference_audio_id", "guidance_scale"}`` -> ``{"id"}``."""
        try:
            body = await request.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            return _error(400, "invalid_json", "The request body is not a JSON object.")
        system_prompt = str(body.get("system_prompt") or "").strip()
        if not system_prompt:
            return _error(400, "prompt_empty", "The system prompt is empty.")
        if len(system_prompt) > SYSTEM_PROMPT_MAX_CHARS:
            return _error(
                400, "prompt_too_long",
                f"The system prompt is longer than {SYSTEM_PROMPT_MAX_CHARS} characters.",
                max_chars=SYSTEM_PROMPT_MAX_CHARS,
            )
        reference = self._reference(str(body.get("reference_audio_id") or ""))
        if reference is None:
            return _error(404, "reference_not_found", "The reference audio was not found; upload or record it again.")
        try:
            guidance_scale = float(body.get("guidance_scale", DEFAULT_GUIDANCE_SCALE))
        except (TypeError, ValueError):
            guidance_scale = DEFAULT_GUIDANCE_SCALE
        if not math.isfinite(guidance_scale):
            guidance_scale = DEFAULT_GUIDANCE_SCALE
        settings = SessionSettings(
            system_prompt=system_prompt,
            reference=reference,
            guidance_scale=min(max(guidance_scale, 1.0), 2.0),  # the trained range
            temperature=self.options.temperature,
            top_k=self.options.top_k,
            max_frames=self.max_frames,
            seed=self.options.seed,
        )
        session_id = uuid.uuid4().hex
        self.sessions[session_id] = settings
        while len(self.sessions) > MAX_STORED:
            self.sessions.popitem(last=False)
        # Run the prompt now if the engine is idle, so the conversation starts at once.
        if self.backend.ready() and not self.slots.locked():
            async with self.slots:
                try:
                    await self._prepare(session_id, settings)
                except Exception:
                    logger.exception("session %s: preparing failed", session_id[:8])
        return web.json_response({"id": session_id})

    # ------------------------------------------------------- conversations

    async def _prepare(self, session_id: str, settings: SessionSettings) -> None:
        await self._discard_prepared()
        conversation = await self.backend.open(settings)
        first = await conversation.start()
        logger.info("session %s: prompt ran in %.2f s", session_id[:8], first.seconds)
        self._prepared = (session_id, conversation, first)

    async def _discard_prepared(self) -> None:
        prepared, self._prepared = self._prepared, None
        if prepared is not None:
            await prepared[1].close()

    async def _begin(self, session_id: str, settings: SessionSettings) -> tuple[Conversation, Frame]:
        """The conversation of ``session_id`` with its frame 0 (call while holding a slot)."""
        if self._prepared is None or self._prepared[0] != session_id:
            await self._prepare(session_id, settings)
        (_, conversation, first), self._prepared = self._prepared, None
        return conversation, first

    async def handle_chat(self, request: web.Request) -> web.WebSocketResponse:
        # A client that stops sending microphone audio must not keep its slot.
        ws = web.WebSocketResponse(receive_timeout=CLIENT_SILENCE_SECONDS)
        await ws.prepare(request)
        session_id = request.query.get("session_id", "")
        settings = self.sessions.get(session_id)

        async def fail(message: str) -> web.WebSocketResponse:
            # The client shows text messages; it has no error message kind.
            await ws.send_bytes(encode_message(KIND_HANDSHAKE))
            await ws.send_bytes(encode_message(KIND_TEXT, f"[error] {message}".encode()))
            await ws.close()
            return ws

        if settings is None:
            return await fail("Session not found; reload the page and try again.")
        if not self.backend.ready():
            return await fail("The model is still loading.")
        async with self.slots:
            if ws.closed:
                return ws
            conversation = None
            try:
                conversation, first = await self._begin(session_id, settings)
                logger.info("session %s: websocket conversation started", session_id[:8])
                frames = await self._converse(ws, conversation, first)
                logger.info("session %s: websocket conversation ended after %d frames", session_id[:8], frames)
            except ConnectionError:
                pass  # the client left
            except asyncio.TimeoutError:
                logger.info("session %s: no microphone audio for %d s, closing", session_id[:8], CLIENT_SILENCE_SECONDS)
            except Exception as error:
                logger.exception("session %s failed", session_id[:8])
                if not ws.closed:
                    await ws.send_bytes(encode_message(KIND_TEXT, f"[error] {error}".encode()))
            finally:
                if conversation is not None:
                    await conversation.close()
        await ws.close()
        return ws

    async def _converse(self, ws: web.WebSocketResponse, conversation: Conversation, first: Frame) -> int:
        """Run one conversation over the WebSocket; returns the number of user frames."""
        reader = sphn.OpusStreamReader(SAMPLE_RATE)
        writer = sphn.OpusStreamWriter(SAMPLE_RATE)
        buffer = FrameBuffer()

        async def send(frame: Frame) -> None:
            pages = writer.append_pcm(frame.audio)
            if pages:
                await ws.send_bytes(encode_message(KIND_AUDIO, pages))
            if frame.text:
                await ws.send_bytes(encode_message(KIND_TEXT, frame.text.encode()))

        async def ping() -> None:
            while not ws.closed:
                await asyncio.sleep(PING_SECONDS)
                await ws.send_bytes(encode_message(KIND_PING))

        await ws.send_bytes(encode_message(KIND_HANDSHAKE))
        await send(first)
        pinger = asyncio.create_task(ping())
        frames = 0
        try:
            async for message in ws:
                if message.type != aiohttp.WSMsgType.BINARY:
                    continue
                kind, payload = decode_message(message.data)
                if kind != KIND_AUDIO or not payload:
                    continue
                pcm = reader.append_bytes(payload)
                if pcm is None or pcm.shape[-1] == 0:
                    continue
                # Input-driven: every complete user frame is answered right away. Audio that
                # arrives in a burst is worked off faster than real time, and the client's
                # playout queue absorbs the burst of replies.
                for user_frame in buffer.push(pcm):
                    frame = await conversation.step(user_frame)
                    frames += 1
                    await send(frame)
                    if frame.done:
                        return frames
        finally:
            pinger.cancel()
        return frames

    # -------------------------------------------------------------- WebRTC

    async def handle_webrtc_config(self, _request: web.Request) -> web.Response:
        """ICE servers for the browser. ``client_turn``: it may bring its own TURN credentials."""
        enabled = self._webrtc_enabled()
        servers = await self.ice.get(self.http) if enabled else []
        return web.json_response(
            {"enabled": enabled, "ice_servers": servers, "client_turn": self.ice is not None},
            headers={"Cache-Control": "no-store"},
        )

    async def handle_webrtc_offer(self, request: web.Request) -> web.Response:
        """Answer a browser's offer; the conversation holds a slot until the peer is gone."""
        if self.ice is None:
            reason = "WebRTC is unavailable: install haru[webrtc]" if webrtc is None else "WebRTC is disabled"
            return web.json_response({"error": reason}, status=404)
        session_id = request.query.get("session_id", "")
        settings = self.sessions.get(session_id)
        if settings is None or not self.backend.ready():
            return web.json_response({"error": "Session not found, or the model is still loading."}, status=404)
        try:
            offer = await request.json()
        except ValueError:
            offer = None
        if not isinstance(offer, dict) or not isinstance(offer.get("sdp"), str) or offer.get("type") != "offer":
            return web.json_response({"error": "invalid offer"}, status=400)
        client_ice = None
        if "ice_servers" in offer:
            client_ice = webrtc.parse_client_ice_servers(offer["ice_servers"])
            if client_ice is None:
                return web.json_response({"error": "ice_servers must be Cloudflare TURN credentials"}, status=400)
        if client_ice is None and not self._webrtc_enabled():
            return web.json_response({"error": "WebRTC is not enabled on this server"}, status=404)

        await self.slots.acquire()
        session = None
        try:
            conversation, first = await self._begin(session_id, settings)
            ice_servers = client_ice or await self.ice.get(self.http)
            groups = {"playout": dict(self.profile["playout"]), "uplink": dict(self.profile["uplink"])}
            for flag, value in self.options.buffers.items():
                group, name = BUFFER_FLAGS[flag]
                groups[group][name] = value
            playout, uplink = PlayoutOptions(**groups["playout"]), UplinkOptions(**groups["uplink"])
            session = webrtc.RTCSession(conversation, first, ice_servers, playout, uplink, FRAME_SAMPLES, SAMPLE_RATE)
            answer = await session.start(offer)
        except Exception as error:
            logger.exception("session %s: WebRTC setup failed", session_id[:8])
            if session is not None:
                await session.close()
            self.slots.release()
            return web.json_response({"error": f"WebRTC setup failed: {error}"}, status=502)

        self.rtc_sessions[session_id] = session
        asyncio.create_task(self._hold_slot(session_id, session))
        return web.json_response(answer)

    async def _hold_slot(self, session_id: str, session) -> None:
        try:
            try:
                await asyncio.wait_for(session.closed.wait(), timeout=RTC_CONNECT_SECONDS)
            except asyncio.TimeoutError:
                if session.pc.connectionState != "connected":
                    await session.close()
                await session.closed.wait()
        finally:
            if self.rtc_sessions.get(session_id) is session:
                del self.rtc_sessions[session_id]
            self.slots.release()
            logger.info("session %s: WebRTC conversation ended: %s", session_id[:8], session.summary())

    async def handle_webrtc_cancel(self, request: web.Request) -> web.Response:
        """The browser gave up on WebRTC: free its slot for the WebSocket fallback."""
        session = self.rtc_sessions.get(request.query.get("session_id", ""))
        if session is not None:
            await session.close()
        return web.json_response({"cancelled": session is not None})

    async def handle_client_stats(self, request: web.Request) -> web.Response:
        """The browser's latency readout, sent every 10 s and at the end of a conversation; only logged."""
        try:
            stats = json.loads((await request.content.read(4096)).decode("utf-8"))
        except ValueError:
            stats = None
        if not isinstance(stats, dict):
            return web.Response(status=400)
        session_id = str(stats.get("session_id") or "")
        if session_id not in self.sessions:
            return web.Response(status=404)

        def ms(key: str) -> str:
            value = stats.get(key)
            return f"{1000 * value:.0f}" if isinstance(value, (int, float)) and math.isfinite(value) else "-"

        transport = stats.get("transport") if stats.get("transport") in ("webrtc", "websocket") else "?"
        logger.info(
            "session %s: client (%s, %s) latency ms p50=%s p95=%s min=%s max=%s | played ms %s, missed ms %s",
            session_id[:8], transport, "final" if stats.get("final") is True else "running",
            ms("latency_p50"), ms("latency_p95"), ms("latency_min"), ms("latency_max"), ms("played"), ms("missed"),
        )
        return web.Response(status=204)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="haru-server", description="Haru web demo server.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", help="checkpoint directory (config.json, model.safetensors, tokenizer.json)")
    source.add_argument("--hf-repo", help=f"Hugging Face repository to download the checkpoint from (default: {DEFAULT_REPO})")
    parser.add_argument("--backend", choices=("local", "vllm"), default="local", help="run the model in this process, or connect to a vLLM-Omni server")
    parser.add_argument("--vllm-url", help="URL of the Haru vLLM-Omni server (with --backend vllm)")
    parser.add_argument("--device", help="torch device of the local backend (default: cuda if available)")
    parser.add_argument("--host", default="localhost", help="address to listen on (0.0.0.0: all interfaces)")
    parser.add_argument("--port", type=int, default=8998)
    parser.add_argument("--static", type=Path, help="directory of a built web client (default: the bundled one)")
    parser.add_argument("--max-sessions", type=int, default=1, help="conversations that may run at the same time; others wait")
    parser.add_argument("--time-limit", type=int, default=3600, help="longest conversation, in seconds")
    parser.add_argument("--language", default="en", help="language preselected in the client on a first visit")
    parser.add_argument("--temperature", type=float, default=0.7, help="text sampling temperature")
    parser.add_argument("--top-k", type=int, default=25, help="text sampling top-k")
    parser.add_argument("--seed", type=int, help="fixed sampling seed for every conversation")
    parser.add_argument("--warmup-frames", type=int, default=16, help="frames of the warm-up conversation after loading (0: none)")
    parser.add_argument(
        "--network-profile", choices=sorted(NETWORK_PROFILES), default="internet",
        help="buffering between browser and server: 'local' (same machine or LAN) keeps thinner buffers",
    )
    buffers = parser.add_argument_group("WebRTC buffers (default: from --network-profile)")
    buffers.add_argument("--start-margin-ms", type=float, help="wait after the first model audio before playback starts")
    buffers.add_argument("--max-buffer-ms", type=float, help="queued model audio above this is dropped")
    buffers.add_argument("--uplink-playout-ms", type=float, help="microphone audio queued before it is fed to the model")
    buffers.add_argument("--uplink-playout-max-ms", type=float, help="limit to which that queue grows when packets arrive late")
    buffers.add_argument("--uplink-shrink-after-s", type=float, help="seconds without a late packet before the queue shrinks again")
    parser.add_argument(
        "--share", action="store_true",
        help="print a public HTTPS link to this server (a tunnel through Gradio's share servers; anyone with the link can use the demo)",
    )
    parser.add_argument(
        "--webrtc", choices=("auto", "on", "off"), default="auto",
        help="offer WebRTC next to the WebSocket: 'auto' when a TURN relay is configured, 'on' also without "
        "(browser and server must reach each other over UDP)",
    )
    parser.add_argument(
        "--ice-servers", metavar="FILE",
        help="JSON list of ICE servers ({urls, username?, credential?}); default: public STUN, or Cloudflare TURN "
        "when CLOUDFLARE_TURN_KEY_ID and CLOUDFLARE_TURN_KEY_API_TOKEN are set in the environment",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("aioice").setLevel(logging.WARNING)  # logs every candidate pair at INFO
    if args.backend == "vllm":
        if not args.vllm_url:
            parser.error("--backend vllm needs --vllm-url")
        backend: Backend = VllmBackend(args.vllm_url)
    else:
        backend = LocalBackend(args.model or args.hf_repo or DEFAULT_REPO, args.device, args.warmup_frames)
    if args.webrtc == "on" and webrtc is None:
        parser.error('--webrtc on needs the optional dependencies: pip install -e "./haru[webrtc]" from the repository')
    options = ServerOptions(
        static_dir=args.static,
        max_sessions=args.max_sessions,
        time_limit_seconds=args.time_limit,
        default_language=args.language,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        network_profile=args.network_profile,
        buffers={flag: getattr(args, flag) for flag in BUFFER_FLAGS if getattr(args, flag) is not None},
        webrtc=args.webrtc,
        ice_servers_file=args.ice_servers,
    )
    server = DemoServer(backend, options)
    app = server.build_app()
    if args.share:
        try:
            from gradio import networking
        except ImportError:
            parser.error('--share needs gradio: pip install -e "./haru[share]" from the repository')

        async def share(_app: web.Application) -> None:
            host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
            link = await asyncio.to_thread(networking.setup_tunnel, host, args.port, secrets.token_urlsafe(32), None, None)
            logger.info("public link: %s", link)

        app.on_startup.append(share)
    logger.info("serving on http://%s:%d (the model is loading)", args.host, args.port)
    web.run_app(app, host=args.host, port=args.port, print=None, access_log=None)


if __name__ == "__main__":
    main()
