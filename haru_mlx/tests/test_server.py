"""The server's API and wire protocol, driven with a stand-in engine (no model needed)."""

import asyncio
import io
import json

import numpy as np
import pytest
import soundfile as sf
import sphn
from aiohttp.test_utils import TestClient, TestServer

from haru_mlx.backends import SessionSettings
from haru_mlx.server import (
    FRAME_SAMPLES,
    KIND_AUDIO,
    KIND_HANDSHAKE,
    KIND_PING,
    KIND_TEXT,
    DemoServer,
    FrameBuffer,
    ServerOptions,
    build_parser,
    decode_message,
    encode_message,
)
from haru_mlx.frames import Frame

SAMPLE_RATE = 24_000


def tone(samples: int) -> np.ndarray:
    return (0.3 * np.sin(2 * np.pi * 220 * np.arange(samples) / SAMPLE_RATE)).astype(np.float32)


class EchoConversation:
    """Answers every user frame with a tone and the frame number as text."""

    def __init__(self, settings: SessionSettings, log: list):
        self.settings = settings
        self.log = log
        self.index = 0

    def _frame(self) -> Frame:
        frame = Frame(tone(FRAME_SAMPLES), token=0, text=f"<{self.index}>", index=self.index, seconds=0.0)
        self.index += 1
        frame.done = self.index >= self.settings.max_frames
        return frame

    async def start(self) -> Frame:
        self.log.append("start")
        return self._frame()

    async def step(self, user_audio: np.ndarray) -> Frame:
        assert user_audio.shape == (FRAME_SAMPLES,) and user_audio.dtype == np.float32
        self.log.append("step")
        return self._frame()

    async def close(self) -> None:
        self.log.append("close")


class EchoBackend:
    def __init__(self):
        self.log: list[str] = []
        self.loaded = False

    async def load(self) -> None:
        self.loaded = True

    def ready(self) -> bool:
        return self.loaded

    async def open(self, settings: SessionSettings) -> EchoConversation:
        return EchoConversation(settings, self.log)


def run(scenario, **options):
    """Run ``scenario(client, server, backend)`` against a live test server."""

    async def main():
        backend = EchoBackend()
        server = DemoServer(backend, ServerOptions(**options))
        # As installed: no default voices; "voice" stands for an uploaded clip.
        server.assets.voices.clear()
        server.assets.voice_files.clear()
        server.references["voice"] = np.zeros(4 * SAMPLE_RATE, dtype=np.float32)
        async with TestClient(TestServer(server.build_app())) as client:
            for _ in range(100):
                if backend.ready():
                    break
                await asyncio.sleep(0.01)
            return await scenario(client, server, backend)

    return asyncio.run(main())


async def create_session(client, **body) -> str:
    body = {"system_prompt": "You are Haru.", "reference_audio_id": "voice", **body}
    response = await client.post("/api/session", json=body)
    assert response.status == 200, await response.text()
    return (await response.json())["id"]


def test_message_framing():
    assert encode_message(KIND_HANDSHAKE) == b"\x00"
    assert encode_message(KIND_TEXT, "はい".encode()) == b"\x02" + "はい".encode()
    assert decode_message(b"\x01abc") == (KIND_AUDIO, b"abc")
    assert decode_message(b"\x06") == (KIND_PING, b"")
    assert decode_message(b"") == (-1, b"")


def test_frame_buffer_cuts_whole_frames():
    buffer = FrameBuffer(4)
    assert buffer.push(np.arange(3)) == []
    frames = buffer.push(np.arange(3, 10))
    assert [f.tolist() for f in frames] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert [f.tolist() for f in buffer.push(np.arange(10, 12))] == [[8, 9, 10, 11]]


def test_config_and_ready():
    async def scenario(client, server, backend):
        assert (await (await client.get("/api/ready")).json()) == {"ready": True}
        config = await (await client.get("/api/config")).json()
        codes = [entry["code"] for entry in config["languages"]]
        assert config["default_language"] == "ja" and "en" in codes
        assert set(config["card_presets"]) == set(codes) == set(config["card_templates"])
        assert config["default_references"] == {}  # no voice ships with the package
        assert config["time_limit_seconds"] == 60 and config["webrtc"] is False
        assert config["playout"]["mic_frames_per_page"] == 1
        assert (await client.get("/api/default_reference?lang=en")).status == 404

    run(scenario, time_limit_seconds=60, network_profile="local", webrtc="off")


def test_reference_upload_trims_and_validates():
    async def scenario(client, server, backend):
        def wav(audio):
            buffer = io.BytesIO()
            sf.write(buffer, audio, 48_000, format="WAV")
            return buffer.getvalue()

        padded = np.concatenate([np.zeros(48_000, np.float32), tone(4 * 48_000), np.zeros(48_000, np.float32)])
        response = await client.post("/api/reference_audio", data=wav(padded))
        info = await response.json()
        assert response.status == 200 and 4.0 <= info["seconds"] <= 4.5 and info["trimmed"] is False
        assert len(server.references[info["id"]]) == round(info["seconds"] * SAMPLE_RATE)
        assert (await client.post("/api/reference_audio", data=wav(tone(48_000)))).status == 400  # too short
        assert (await client.post("/api/reference_audio", data=b"")).status == 400
        assert (await client.post("/api/reference_audio", data=b"not audio")).status == 400
        session_id = await create_session(client, reference_audio_id=info["id"])
        assert server.sessions[session_id].reference is server.references[info["id"]]

    run(scenario)


def test_session_validation_and_guidance_clamp():
    async def scenario(client, server, backend):
        assert (await client.post("/api/session", data=b"{")).status == 400
        assert (await client.post("/api/session", json={"system_prompt": " ", "reference_audio_id": "voice"})).status == 400
        assert (await client.post("/api/session", json={"system_prompt": "x" * 4001, "reference_audio_id": "voice"})).status == 400
        assert (await client.post("/api/session", json={"system_prompt": "x", "reference_audio_id": "nope"})).status == 404
        assert server.sessions[await create_session(client)].guidance_scale == 1.5
        assert server.sessions[await create_session(client, guidance_scale=9)].guidance_scale == 2.0
        assert server.sessions[await create_session(client, guidance_scale="x")].guidance_scale == 1.5
        settings = server.sessions[await create_session(client)]
        assert settings.reference is server.references["voice"]
        assert settings.max_frames == 3600 * 24_000 // 3200

    run(scenario)


def test_chat_streams_one_model_frame_ahead():
    async def scenario(client, server, backend):
        session_id = await create_session(client)
        assert backend.log == ["start"]  # the prompt ran when the session was registered
        ws = await client.ws_connect(f"/api/chat?session_id={session_id}")
        reader, writer = sphn.OpusStreamReader(SAMPLE_RATE), sphn.OpusStreamWriter(SAMPLE_RATE)
        assert decode_message((await ws.receive()).data) == (KIND_HANDSHAKE, b"")
        texts, samples = [], 0
        for _ in range(6):
            await ws.send_bytes(encode_message(KIND_AUDIO, writer.append_pcm(tone(FRAME_SAMPLES))))
        # Opus delays its output a little: a short tail pushes the sixth frame through.
        await ws.send_bytes(encode_message(KIND_AUDIO, writer.append_pcm(np.zeros(1920, np.float32))))
        while len(texts) < 7:
            kind, payload = decode_message((await asyncio.wait_for(ws.receive(), 5)).data)
            if kind == KIND_AUDIO:
                pcm = reader.append_bytes(payload)
                samples += 0 if pcm is None else pcm.shape[-1]
            elif kind == KIND_TEXT:
                texts.append(payload.decode())
        await ws.close()
        for _ in range(100):
            if backend.log[-1] == "close":
                break
            await asyncio.sleep(0.01)
        assert texts == [f"<{i}>" for i in range(7)]  # frame 0 unprompted, then one per user frame
        assert samples >= 6 * FRAME_SAMPLES
        assert backend.log == ["start"] + ["step"] * 6 + ["close"]

    run(scenario)


def test_chat_ends_at_the_time_limit_and_queues_the_next_conversation():
    async def scenario(client, server, backend):
        first, second = await create_session(client), await create_session(client)
        ws1 = await client.ws_connect(f"/api/chat?session_id={first}")
        assert decode_message((await ws1.receive()).data)[0] == KIND_HANDSHAKE
        ws2 = await client.ws_connect(f"/api/chat?session_id={second}")
        with pytest.raises(asyncio.TimeoutError):  # one conversation at a time
            await asyncio.wait_for(ws2.receive(), 0.3)
        writer = sphn.OpusStreamWriter(SAMPLE_RATE)
        for _ in range(10):  # a 1 s limit is 7 model frames: frame 0 and six replies
            await ws1.send_bytes(encode_message(KIND_AUDIO, writer.append_pcm(tone(FRAME_SAMPLES))))
        async for _ in ws1:
            pass
        assert ws1.closed and backend.log.count("step") == 6
        assert decode_message((await asyncio.wait_for(ws2.receive(), 5)).data)[0] == KIND_HANDSHAKE
        await ws2.close()

    run(scenario, time_limit_seconds=1)


def test_chat_reports_an_unknown_session_as_text():
    async def scenario(client, server, backend):
        ws = await client.ws_connect("/api/chat?session_id=missing")
        assert decode_message((await ws.receive()).data)[0] == KIND_HANDSHAKE
        kind, payload = decode_message((await ws.receive()).data)
        assert kind == KIND_TEXT and payload.decode().startswith("[error]")

    run(scenario)


def test_webrtc_is_off_without_a_relay():
    async def scenario(client, server, backend):
        config = await (await client.get("/api/webrtc/config")).json()
        assert config["enabled"] is False and config["ice_servers"] == []
        session_id = await create_session(client)
        response = await client.post(f"/api/webrtc/offer?session_id={session_id}", json={"sdp": "v=0", "type": "offer"})
        assert response.status == 404 and "error" in await response.json()
        assert (await (await client.post("/api/webrtc/cancel?session_id=x")).json()) == {"cancelled": False}

    run(scenario)


def test_webrtc_routes_report_missing_dependencies(monkeypatch):
    monkeypatch.setattr("haru_mlx.server.webrtc", None)  # as when aiortc is not installed

    async def scenario(client, server, backend):
        config = await (await client.get("/api/webrtc/config")).json()
        assert config == {"enabled": False, "ice_servers": [], "client_turn": False}
        assert (await (await client.get("/api/config")).json())["webrtc"] is False
        session_id = await create_session(client)
        response = await client.post(f"/api/webrtc/offer?session_id={session_id}", json={"sdp": "v=0", "type": "offer"})
        assert response.status == 404 and "unavailable" in (await response.json())["error"]

    run(scenario, webrtc="auto")


def test_client_stats_are_accepted_for_known_sessions():
    async def scenario(client, server, backend):
        session_id = await create_session(client)
        stats = {"session_id": session_id, "final": True, "transport": "websocket", "latency_p50": 0.31}
        assert (await client.post("/api/client_stats", data=json.dumps(stats))).status == 204
        assert (await client.post("/api/client_stats", data=json.dumps({"session_id": "x"}))).status == 404
        assert (await client.post("/api/client_stats", data="[]")).status == 400

    run(scenario)


def test_static_client(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<html>haru</html>")
    (tmp_path / "assets" / "app.js").write_text("export {};")

    async def scenario(client, server, backend):
        index = await client.get("/")
        assert index.status == 200 and index.content_type == "text/html" and "haru" in await index.text()
        script = await client.get("/assets/app.js")
        assert script.status == 200 and "javascript" in script.content_type
        assert (await client.get("/assets/missing.js")).status == 404
        assert (await client.get("/assets/../index.html")).status == 404

    run(scenario, static_dir=tmp_path)


def test_command_line():
    args = build_parser().parse_args(["--hf-repo", "higurashi-ai/Haru-4B"])
    assert (args.port, args.quantize, args.max_sessions, args.webrtc) == (8998, "8", 1, "auto")
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--model", "a", "--hf-repo", "b"])
