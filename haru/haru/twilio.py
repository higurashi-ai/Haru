"""Twilio Media Streams audio and protocol helpers.

Twilio sends and receives base64-encoded G.711 mu-law at 8 kHz. Haru consumes
and produces 3200-sample float32 frames at 24 kHz. This module keeps both the
codec and resampler state for one WebSocket, so arbitrary Media Streams packet
boundaries do not introduce clicks or timing drift.

Protocol: https://www.twilio.com/docs/voice/media-streams/websocket-messages
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import firwin, lfilter

TELEPHONY_SAMPLE_RATE = 8_000
MODEL_SAMPLE_RATE = 24_000
MODEL_FRAME_SAMPLES = 3_200
MEDIA_PACKET_SAMPLES = 160  # 20 ms at 8 kHz


def _decode_mulaw_byte(value: int) -> int:
    value = (~value) & 0xFF
    sample = ((value & 0x0F) << 3) + 0x84
    sample <<= (value & 0x70) >> 4
    return 0x84 - sample if value & 0x80 else sample - 0x84


def _encode_mulaw_sample(value: int) -> int:
    value = max(-32768, min(32767, value))
    sign = (value >> 8) & 0x80
    if sign:
        value = -value
    value = min(value, 32635) + 0x84
    exponent = 7
    mask = 0x4000
    while not value & mask and exponent:
        exponent -= 1
        mask >>= 1
    mantissa = (value >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa)) & 0xFF


_MULAW_DECODE = np.asarray([_decode_mulaw_byte(i) for i in range(256)], dtype=np.int16)
_MULAW_ENCODE = np.asarray([_encode_mulaw_sample(i - 32768) for i in range(65536)], dtype=np.uint8)


def decode_mulaw(data: bytes) -> np.ndarray:
    """G.711 mu-law bytes to mono float32 PCM in ``[-1, 1)``."""
    encoded = np.frombuffer(data, dtype=np.uint8)
    return (_MULAW_DECODE[encoded].astype(np.float32) / 32768.0).reshape(-1)


def encode_mulaw(audio: np.ndarray) -> bytes:
    """Mono float PCM to G.711 mu-law bytes."""
    pcm = np.rint(np.clip(np.asarray(audio, dtype=np.float32), -1.0, 32767 / 32768) * 32768).astype(np.int16)
    return _MULAW_ENCODE[pcm.astype(np.int32) + 32768].tobytes()


class StreamingResampler:
    """A causal rational resampler with state across pushes.

    A Kaiser-window FIR removes images and aliases. Persistent filter and
    decimation state makes output independent of packet boundaries.
    """

    def __init__(self, source_rate: int, target_rate: int):
        divisor = math.gcd(source_rate, target_rate)
        self.up = target_rate // divisor
        self.down = source_rate // divisor
        width = max(self.up, self.down)
        taps = firwin(20 * width + 1, 1.0 / width, window=("kaiser", 5.0)).astype(np.float64)
        self.taps = taps * self.up
        self.state = np.zeros(len(self.taps) - 1, dtype=np.float64)
        self.processed = 0

    def push(self, audio: np.ndarray) -> np.ndarray:
        source = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not len(source):
            return np.zeros(0, dtype=np.float32)
        if self.up == 1:
            expanded = source.astype(np.float64)
        else:
            expanded = np.zeros(len(source) * self.up, dtype=np.float64)
            expanded[:: self.up] = source
        filtered, self.state = lfilter(self.taps, [1.0], expanded, zi=self.state)
        offset = (-self.processed) % self.down
        self.processed += len(expanded)
        return filtered[offset:: self.down].astype(np.float32)


@dataclass(frozen=True)
class Start:
    stream_sid: str
    session_id: str


def parse_start(message: Any) -> Start | None:
    """Return the fields Haru needs from a Twilio ``start`` message."""
    if not isinstance(message, dict) or message.get("event") != "start":
        return None
    start = message.get("start")
    if not isinstance(start, dict):
        return None
    stream_sid = start.get("streamSid") or message.get("streamSid")
    parameters = start.get("customParameters")
    session_id = parameters.get("session_id") if isinstance(parameters, dict) else None
    media_format = start.get("mediaFormat")
    valid_format = isinstance(media_format, dict) and (
        media_format.get("encoding") == "audio/x-mulaw"
        and media_format.get("sampleRate") == TELEPHONY_SAMPLE_RATE
        and media_format.get("channels") == 1
    )
    if (
        not valid_format
        or not isinstance(stream_sid, str)
        or not stream_sid
        or not isinstance(session_id, str)
        or not session_id
    ):
        return None
    return Start(stream_sid=stream_sid, session_id=session_id)


class InputAudio:
    """Media Streams payloads to whole Haru input frames."""

    def __init__(self, frame_samples: int = MODEL_FRAME_SAMPLES):
        self.resampler = StreamingResampler(TELEPHONY_SAMPLE_RATE, MODEL_SAMPLE_RATE)
        self.frame_samples = frame_samples
        self.pending = np.zeros(0, dtype=np.float32)

    def push(self, payload: str) -> list[np.ndarray]:
        raw = base64.b64decode(payload, validate=True)
        converted = self.resampler.push(decode_mulaw(raw))
        self.pending = np.concatenate((self.pending, converted))
        count = len(self.pending) // self.frame_samples
        frames = [
            self.pending[index * self.frame_samples : (index + 1) * self.frame_samples]
            for index in range(count)
        ]
        self.pending = self.pending[count * self.frame_samples :]
        return frames


class OutputAudio:
    """Haru output frames to fixed 20 ms Media Streams messages."""

    def __init__(self, stream_sid: str, packet_samples: int = MEDIA_PACKET_SAMPLES):
        self.stream_sid = stream_sid
        self.packet_samples = packet_samples
        self.resampler = StreamingResampler(MODEL_SAMPLE_RATE, TELEPHONY_SAMPLE_RATE)
        self.pending = np.zeros(0, dtype=np.float32)

    def push(self, audio: np.ndarray) -> list[dict[str, Any]]:
        self.pending = np.concatenate((self.pending, self.resampler.push(audio)))
        count = len(self.pending) // self.packet_samples
        messages = []
        for index in range(count):
            packet = self.pending[index * self.packet_samples : (index + 1) * self.packet_samples]
            messages.append(
                {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": base64.b64encode(encode_mulaw(packet)).decode("ascii")},
                }
            )
        self.pending = self.pending[count * self.packet_samples :]
        return messages
