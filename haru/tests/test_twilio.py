"""Twilio Media Streams transport, without a model or telephone network."""

import asyncio
import base64
import json

import numpy as np
from aiohttp.test_utils import TestClient, TestServer
from test_server import SAMPLE_RATE, EchoBackend, create_session

from haru import twilio
from haru.server import DemoServer, ServerOptions


def test_mulaw_known_values_and_round_trip():
    assert twilio.encode_mulaw(np.zeros(4, dtype=np.float32)) == b"\xff" * 4
    decoded = twilio.decode_mulaw(bytes(range(256)))
    assert decoded.dtype == np.float32 and decoded.shape == (256,)
    assert np.all(np.isfinite(decoded))
    source = np.linspace(-0.95, 0.95, 1000, dtype=np.float32)
    restored = twilio.decode_mulaw(twilio.encode_mulaw(source))
    assert np.max(np.abs(restored - source)) < 0.035


def test_streaming_resampler_is_independent_of_packet_boundaries():
    source = np.sin(2 * np.pi * 440 * np.arange(1600) / 8000).astype(np.float32)
    whole = twilio.StreamingResampler(8000, 24000).push(source)
    split_resampler = twilio.StreamingResampler(8000, 24000)
    split = np.concatenate([split_resampler.push(part) for part in np.array_split(source, [13, 160, 777])])
    assert len(whole) == len(split) == 4800
    assert np.array_equal(whole, split)

    down = twilio.StreamingResampler(24000, 8000)
    chunks = [down.push(np.ones(3200, np.float32)) for _ in range(3)]
    assert [len(chunk) for chunk in chunks] == [1067, 1067, 1066]


def test_audio_framing_and_protocol():
    start = twilio.parse_start(
        {
            "event": "start",
            "streamSid": "MZouter",
            "start": {
                "streamSid": "MZinner",
                "customParameters": {"session_id": "session"},
                "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
            },
        }
    )
    assert start == twilio.Start(stream_sid="MZinner", session_id="session")
    assert twilio.parse_start({"event": "start", "start": {"customParameters": {}}}) is None

    incoming = twilio.InputAudio()
    silence = base64.b64encode(b"\xff" * 160).decode()
    assert [len(incoming.push(silence)) for _ in range(6)] == [0] * 6
    assert [len(frame) for frame in incoming.push(silence)] == [3200]

    outgoing = twilio.OutputAudio("MZtest")
    first = outgoing.push(np.zeros(3200, np.float32))
    second = outgoing.push(np.zeros(3200, np.float32))
    third = outgoing.push(np.zeros(3200, np.float32))
    assert [len(first), len(second), len(third)] == [6, 7, 7]
    assert all(message["event"] == "media" and message["streamSid"] == "MZtest" for message in first + second + third)
    payloads = [base64.b64decode(message["media"]["payload"]) for message in first + second + third]
    assert all(len(payload) == 160 for payload in payloads)


def test_server_twilio_transport():
    async def main():
        backend = EchoBackend()
        server = DemoServer(backend, ServerOptions(webrtc="off"))
        server.assets.voices.clear()
        server.assets.voice_files.clear()
        server.references["voice"] = np.zeros(4 * SAMPLE_RATE, dtype=np.float32)
        async with TestClient(TestServer(server.build_app())) as client:
            for _ in range(100):
                if backend.ready():
                    break
                await asyncio.sleep(0.01)
            session_id = await create_session(client)
            ws = await client.ws_connect("/api/twilio")
            await ws.send_json({"event": "connected", "protocol": "Call", "version": "1.0.0"})
            await ws.send_json(
                {
                    "event": "start",
                    "streamSid": "MZtest",
                    "start": {
                        "streamSid": "MZtest",
                        "customParameters": {"session_id": session_id},
                        "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
                    },
                }
            )
            initial = [json.loads((await asyncio.wait_for(ws.receive(), 2)).data) for _ in range(6)]
            assert all(message["event"] == "media" and message["streamSid"] == "MZtest" for message in initial)

            payload = base64.b64encode(b"\xff" * 160).decode()
            for _ in range(7):
                await ws.send_json({"event": "media", "streamSid": "MZtest", "media": {"payload": payload}})
            reply = [json.loads((await asyncio.wait_for(ws.receive(), 2)).data) for _ in range(7)]
            assert all(len(base64.b64decode(message["media"]["payload"])) == 160 for message in reply)
            await ws.send_json({"event": "stop", "streamSid": "MZtest", "stop": {}})
            await ws.close()
            for _ in range(100):
                if backend.log[-1] == "close":
                    break
                await asyncio.sleep(0.01)
            assert backend.log == ["start", "step", "close"]

    asyncio.run(main())
