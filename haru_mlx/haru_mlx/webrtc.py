"""WebRTC transport of the web demo (optional: ``pip install "haru_mlx[webrtc]"``).

The browser sends its microphone as an Opus track and plays the model's track;
text and session events travel on the "events" DataChannel. Over a lossy or
jittery network this beats the WebSocket path: a late or lost packet is
concealed instead of stalling everything behind it.

    uplink:   each decoded 20 ms of microphone audio is appended to the user
              stream (gaps in the RTP timestamps become silence); the model
              steps whenever 133 ms are complete
    downlink: 133 ms model frames, paced out as 20 ms packets by a
              :class:`~haru_mlx.playout.PlayoutBuffer`

Importing this module needs aiortc; the server works without it and then
reports WebRTC as unavailable.
"""

from __future__ import annotations

import asyncio
import fractions
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
from aiortc import MediaStreamTrack, RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
from aiortc import rtcrtpreceiver
from aiortc.jitterbuffer import JitterFrame
from aiortc.mediastreams import MediaStreamError
from aiortc.utils import uint16_gt
from av import AudioFrame

from .backends import Conversation
from .playout import PACKET_SAMPLES, RTC_SAMPLE_RATE, PlayoutBuffer, PlayoutOptions, UplinkBuffer, UplinkOptions
from .session import Frame

logger = logging.getLogger(__name__)

EVENTS_CHANNEL_ID = 4  # the pre-negotiated "events" DataChannel (the offer's ``dcid``)
PUBLIC_STUN_SERVERS = [{"urls": ["stun:stun.l.google.com:19302", "stun:stun.cloudflare.com:3478"]}]
_CLOUDFLARE_TURN_URL = "https://rtc.live.cloudflare.com/v1/turn/keys/{key_id}/credentials/generate-ice-servers"
_CLOUDFLARE_ICE_URL = re.compile(r"^(stun|turns?):([a-z0-9.-]+\.)?cloudflare\.com(:\d+)?(\?transport=(udp|tcp))?$")


# ---------------------------------------------------------------- ICE servers


class IceServers:
    """ICE servers for both peers, in the browser's ``RTCIceServer`` JSON form.

    A static list (``--ice-servers`` file, else public STUN), or short-lived
    Cloudflare TURN credentials when the environment holds a TURN key
    (``CLOUDFLARE_TURN_KEY_ID`` and ``CLOUDFLARE_TURN_KEY_API_TOKEN``). The key
    stays on the server; browsers only receive the generated credentials.
    """

    def __init__(self, path: str | None = None, ttl_seconds: int = 7200, conversation_seconds: int = 3600):
        self.static = json.loads(Path(path).read_text(encoding="utf-8")) if path else PUBLIC_STUN_SERVERS
        self.key_id = os.environ.get("CLOUDFLARE_TURN_KEY_ID", "")
        self.token = os.environ.get("CLOUDFLARE_TURN_KEY_API_TOKEN", "")
        self.ttl_seconds = ttl_seconds
        # Refresh early enough that a whole conversation fits into the credentials' lifetime.
        self._lifetime = max(60.0, ttl_seconds - conversation_seconds - 60)
        self._generated: list[dict[str, Any]] = []
        self._expires = 0.0
        self._lock = asyncio.Lock()

    @property
    def has_turn(self) -> bool:
        """Whether a relay is configured (needed when the peers cannot reach each other directly)."""
        return bool(self.key_id and self.token) or any("username" in server for server in self.static)

    async def get(self, http: aiohttp.ClientSession) -> list[dict[str, Any]]:
        if not (self.key_id and self.token):
            return self.static
        async with self._lock:
            if self._generated and time.monotonic() < self._expires:
                return self._generated
            try:
                url = _CLOUDFLARE_TURN_URL.format(key_id=self.key_id)
                headers = {"Authorization": f"Bearer {self.token}"}
                async with http.post(url, headers=headers, json={"ttl": self.ttl_seconds}) as response:
                    response.raise_for_status()
                    payload = await response.json(content_type=None)
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as error:
                # The error type only: a message could echo request details.
                logger.warning("TURN credentials request failed: %s", type(error).__name__)
                return self._generated or self.static
            self._generated = _normalize_ice_servers(payload)
            self._expires = time.monotonic() + self._lifetime
            return self._generated


def _normalize_ice_servers(payload: Any) -> list[dict[str, Any]]:
    servers = payload.get("iceServers", payload) if isinstance(payload, dict) else payload
    servers = [servers] if isinstance(servers, dict) else servers or []
    out = []
    for server in servers:
        urls = server.get("urls") or []
        urls = [urls] if isinstance(urls, str) else list(urls)
        urls = [u for u in urls if not re.search(r":53(\?|$)", u)]  # browsers block port 53
        if urls:
            out.append({"urls": urls, **{k: server[k] for k in ("username", "credential") if server.get(k)}})
    return out


def parse_client_ice_servers(raw: Any) -> list[dict[str, Any]] | None:
    """Validate TURN credentials a browser generated from the visitor's own Cloudflare key.

    Only Cloudflare's hosts are accepted, so a client cannot point the server's
    ICE agent at an arbitrary address. ``None`` when malformed.
    """
    if not isinstance(raw, list) or not 0 < len(raw) <= 4:
        return None
    out = []
    for server in raw:
        if not isinstance(server, dict):
            return None
        urls = server.get("urls")
        urls = [urls] if isinstance(urls, str) else urls
        if not isinstance(urls, list) or not 0 < len(urls) <= 8:
            return None
        if not all(isinstance(u, str) and len(u) <= 256 and _CLOUDFLARE_ICE_URL.match(u) for u in urls):
            return None
        entry: dict[str, Any] = {"urls": urls}
        for key in ("username", "credential"):
            value = server.get(key)
            if value is not None:
                if not isinstance(value, str) or len(value) > 1024:
                    return None
                entry[key] = value
        out.append(entry)
    return out if any("username" in entry for entry in out) else None


# ------------------------------------------------------------------- media


class _PassThroughJitterBuffer:
    """Hands every audio packet to the decoder as it arrives; late ones are dropped."""

    def __init__(self) -> None:
        self._last: int | None = None

    def add(self, packet):
        if self._last is not None and uint16_gt(self._last, packet.sequence_number):
            return False, None
        self._last = packet.sequence_number
        return False, JitterFrame(data=packet._data, timestamp=packet.timestamp)


_aiortc_jitter_buffer = rtcrtpreceiver.JitterBuffer


def _jitter_buffer(capacity: int, prefetch: int = 0, is_video: bool = False):
    if is_video:
        return _aiortc_jitter_buffer(capacity=capacity, prefetch=prefetch, is_video=True)
    return _PassThroughJitterBuffer()


# aiortc's audio jitter buffer holds back 4 packets (80 ms) and, after a lost
# packet, waits for 16 more before discarding the backlog. The session fills
# gaps itself, so incoming audio skips that buffer.
rtcrtpreceiver.JitterBuffer = _jitter_buffer


class ModelAudioTrack(MediaStreamTrack):
    """The model's voice: a :class:`PlayoutBuffer` read on a 20 ms clock."""

    kind = "audio"

    def __init__(self, options: PlayoutOptions):
        super().__init__()
        self.buffer = PlayoutBuffer(options)
        self._first_audio: float | None = None
        self._clock: float | None = None
        self._pts = 0

    def push(self, pcm24: np.ndarray) -> None:
        self.buffer.push(pcm24)
        if self._first_audio is None:
            self._first_audio = time.monotonic()

    async def recv(self) -> AudioFrame:
        if self.readyState != "live":
            raise MediaStreamError
        now = time.monotonic()
        if self._clock is None:
            self._clock = now
        else:
            wait = self._clock + self._pts / RTC_SAMPLE_RATE - now
            if wait > 0:
                await asyncio.sleep(wait)
        margin = self.buffer.options.start_margin_ms / 1000
        if not self.buffer.started and self._first_audio is not None and time.monotonic() >= self._first_audio + margin:
            self.buffer.start()
        frame = AudioFrame.from_ndarray(self.buffer.pop().reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = RTC_SAMPLE_RATE
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, RTC_SAMPLE_RATE)
        self._pts += PACKET_SAMPLES
        return frame


def _to_model_rate(frame: AudioFrame) -> np.ndarray:
    """A decoded 48 kHz frame -> mono float32 at 24 kHz.

    Pairwise means instead of a stateful resampler: every 20 ms maps to exactly
    480 samples, so the user's timeline never drifts.
    """
    x = frame.to_ndarray().astype(np.float32).reshape(-1)
    channels = len(frame.layout.channels)
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return x[: len(x) // 2 * 2].reshape(-1, 2).mean(axis=1) / 32768.0


class RTCSession:
    """One browser peer connection driving one :class:`Conversation`."""

    def __init__(
        self,
        conversation: Conversation,
        first_frame: Frame,
        ice_servers: list[dict[str, Any]],
        playout: PlayoutOptions,
        uplink: UplinkOptions | None = None,
        frame_samples: int = 3200,
        sample_rate: int = 24_000,
    ):
        self.conversation = conversation
        self.frame_samples = frame_samples
        self.sample_rate = sample_rate
        servers = [RTCIceServer(urls=s["urls"], username=s.get("username"), credential=s.get("credential")) for s in ice_servers]
        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=servers))
        self.track = ModelAudioTrack(playout)
        self.uplink = UplinkBuffer(uplink, slot_samples=sample_rate // 50)
        self._feeder: asyncio.Task | None = None
        self.channel = None
        self.closed = asyncio.Event()
        self.frames = 0
        self.user_ms = 0.0
        self._pending: list[str] = []  # events produced before the DataChannel opened
        self._user_frames: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []
        self._lag: list[float] = []
        self.send_event({"type": "session.ready"})
        self._emit(first_frame)

    def send_event(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False)
        if self.channel is not None and self.channel.readyState == "open":
            self.channel.send(data)
        elif len(self._pending) < 1024:
            self._pending.append(data)

    def _emit(self, frame: Frame) -> None:
        self.track.push(frame.audio)
        if frame.text:
            self.send_event({"type": "text", "text": frame.text})

    async def start(self, offer: dict[str, Any]) -> dict[str, str]:
        """Answer the browser's offer ``{"sdp", "type", "dcid"}``."""
        dcid = offer.get("dcid", EVENTS_CHANNEL_ID)
        if not isinstance(dcid, int) or isinstance(dcid, bool) or not 0 <= dcid < 1024:
            raise ValueError("dcid must be a DataChannel id")
        # Both peers create the channel with the same id: no in-band open handshake.
        self.channel = self.pc.createDataChannel("events", negotiated=True, id=dcid)

        @self.channel.on("open")
        def on_open() -> None:
            for data in self._pending:
                self.channel.send(data)
            self._pending.clear()

        @self.channel.on("message")
        def on_message(message) -> None:
            try:
                ended = json.loads(message).get("type") == "session.end"
            except (TypeError, ValueError, AttributeError):
                return
            if ended:
                asyncio.ensure_future(self.close())

        @self.pc.on("track")
        def on_track(track) -> None:
            if track.kind == "audio":
                self._tasks.append(asyncio.ensure_future(self._uplink(track)))

        @self.pc.on("connectionstatechange")
        async def on_state() -> None:
            if self.pc.connectionState in ("failed", "closed", "disconnected"):
                await self.close()

        self.pc.addTrack(self.track)
        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=offer["sdp"], type=offer["type"]))
        await self.pc.setLocalDescription(await self.pc.createAnswer())
        self._tasks.append(asyncio.ensure_future(self._generate()))
        self._tasks.append(asyncio.ensure_future(self._report()))
        return {"sdp": self.pc.localDescription.sdp, "type": self.pc.localDescription.type}

    def lag_ms(self) -> float:
        """User stream time received minus model stream time sent (skipped audio counts as sent)."""
        buffer = self.track.buffer
        return self.user_ms - buffer.sent_ms - buffer.dropped_ms - buffer.compressed_ms

    async def _uplink(self, track) -> None:
        """Microphone packets into the jitter buffer, as 20 ms slots at the model's sample rate."""
        slot = self.sample_rate // 50
        try:
            while not self.closed.is_set():
                frame = await track.recv()
                if frame.pts is None or frame.sample_rate != RTC_SAMPLE_RATE:
                    continue
                pcm = _to_model_rate(frame)
                packets = [pcm[i : i + slot] for i in range(0, len(pcm) - slot + 1, slot)]
                self.uplink.add(int(frame.pts) // PACKET_SAMPLES, packets, time.monotonic())
                if self._feeder is None:
                    self._feeder = asyncio.ensure_future(self._feed())
                    self._tasks.append(self._feeder)
        except (MediaStreamError, ConnectionError, asyncio.CancelledError):
            pass
        finally:
            await self.close()

    async def _feed(self) -> None:
        """One slot of user audio per 20 ms of wall clock, grouped into whole model frames."""
        while not self.uplink.ready:
            await asyncio.sleep(0.005)
        pending: list[np.ndarray] = []
        queued = 0
        start, n = time.monotonic(), 0
        while not self.closed.is_set():
            wait = start + 0.020 * n - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            n += 1
            pcm = self.uplink.take(time.monotonic())
            self.user_ms += 20.0
            if self.track.buffer.started:
                self._lag.append(self.lag_ms())
            pending.append(pcm)
            queued += len(pcm)
            if queued >= self.frame_samples:
                audio = np.concatenate(pending)
                self._user_frames.put_nowait(audio[: self.frame_samples])
                rest = audio[self.frame_samples :]
                pending, queued = ([rest], len(rest)) if len(rest) else ([], 0)

    async def _generate(self) -> None:
        try:
            while True:
                frame = await self.conversation.step(await self._user_frames.get())
                self.frames += 1
                self._emit(frame)
                if frame.done:
                    self.send_event({"type": "session.done"})
                    break
        except asyncio.CancelledError:
            return
        except Exception as error:
            logger.exception("conversation failed")
            self.send_event({"type": "error", "message": str(error)})
        await self.close()

    async def _report(self) -> None:
        """Once a second: the server's part of the mouth-to-ear delay, for the client's readout."""
        while not self.closed.is_set():
            await asyncio.sleep(1.0)
            if self._lag:
                self.send_event({"type": "latency", "lag_ms": round(float(np.mean(self._lag)), 1)})
                self._lag.clear()

    def summary(self) -> str:
        buffer = self.track.buffer
        return (
            f"{self.frames} frames, user audio {self.user_ms / 1000:.1f} s, model audio sent {buffer.sent_ms / 1000:.1f} s, "
            f"underrun {buffer.underrun_ms:.0f} ms, dropped {buffer.dropped_ms:.0f} ms, "
            f"catch-up {buffer.compressed_ms:.0f} ms; user audio lost {self.uplink.lost_ms:.0f} ms, "
            f"concealed {self.uplink.concealed_ms:.0f} ms, late {self.uplink.late_ms:.0f} ms, "
            f"trimmed {self.uplink.trimmed_ms:.0f} ms, uplink delay {self.uplink.target_ms:.0f} ms"
        )

    async def close(self) -> None:
        if self.closed.is_set():
            return
        self.closed.set()
        current = asyncio.current_task()
        for task in self._tasks:
            if task is not current:
                task.cancel()
        self.track.stop()
        await self.pc.close()
        await self.conversation.close()
