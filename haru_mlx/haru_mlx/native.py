"""Local socket server for native clients that do their own audio I/O (the macOS app).

    python -m haru_mlx.native [--socket PATH]

The client owns the microphone and the speakers; this process runs the model.
One client at a time, over a Unix socket on the same machine, so the transport
adds well under a millisecond and needs no jitter buffer. A message is a type
byte, a little-endian u32 payload length, then the payload:

    server -> client
        0x13 ASSETS  JSON, on connect: card templates ("templates"), example
                     cards ("presets"), default voice files ("voices"),
                     "languages", "default_language", "step_ms", "precision",
                     "model"
        0x10 HELLO   JSON: sample_rate, frame_samples, step_ms
        0x11 FRAME   u32 frame index, f32 step ms, float32 mono PCM at 48 kHz
        0x12 TEXT    UTF-8 text of the model
        0x14 ERROR   JSON: message
        0x15 READY   JSON: seq (the PREPARE it answers, from 1), ok, message
    client -> server
        0x03 PREPARE JSON, any number before START: the same fields as START
        0x02 START   JSON, once: any of "card" + "language" (rendered with the
                     templates) or "system_prompt", "reference_path" (an audio
                     file on this machine), "guidance_scale", "speak_first";
                     missing fields use the server's defaults
        0x01 AUDIO   float32 mono PCM at 48 kHz, any length

PREPARE runs the prompt of that request in the background (a newer PREPARE
replaces an older one) and answers READY. START with the same request as the
last PREPARE then answers at once; otherwise the server runs the prompt first.
After that the server sends HELLO and model frame 0, and from then on model
frame ``k + 1`` as soon as the client's user frame ``k`` is complete. When to
play a frame (the start, slack for a late step, catching up) is the client's
job.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import struct
import tempfile
import time
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from .audio import load_audio, prepare_reference
from .backends import LocalBackend, SessionSettings
from .frames import Frame
from .hub import DEFAULT_REPO
from .prompt import card_templates, render_card

logger = logging.getLogger("haru_mlx.native")

CLIENT_SAMPLE_RATE = 48_000
AUDIO, START, PREPARE = 0x01, 0x02, 0x03
HELLO, FRAME, TEXT, ASSETS, ERROR, READY = 0x10, 0x11, 0x12, 0x13, 0x14, 0x15
REFERENCE_MIN_SECONDS = 3.0
PRESET_LABELS = {"chat": "初対面の雑談"}
STEP_ALLOWANCE_SECONDS = 0.003  # encoding the last audio block of a frame, on top of the step


def default_socket() -> str:
    """The server's socket, in this user's temporary directory."""
    return os.path.join(tempfile.gettempdir(), "haru.sock")


def upsample_2x(pcm: np.ndarray) -> np.ndarray:
    """24 kHz -> 48 kHz by linear interpolation (enough for speech playback)."""
    x = np.asarray(pcm, dtype=np.float32)
    up = np.empty(2 * len(x), dtype=np.float32)
    up[0::2] = x
    up[1::2] = np.concatenate([(x[:-1] + x[1:]) * 0.5, x[-1:]])
    return up


def downsample_2x(pcm: np.ndarray) -> np.ndarray:
    """48 kHz -> 24 kHz by pairwise means: every two samples map to exactly one."""
    return pcm.reshape(-1, 2).mean(axis=1)


def bundled_assets() -> dict[str, Any]:
    """Card templates, example cards and the default voice files that exist."""
    root = resources.files("haru_mlx") / "assets"
    templates = card_templates()
    codes = [entry["code"] for entry in templates["languages"]]
    presets = {}
    for code in codes:
        cards = json.loads((root / "cards" / f"{code}.json").read_text(encoding="utf-8"))
        presets[code] = [{"id": key, "label": PRESET_LABELS.get(key, key), "card": card} for key, card in cards.items()]
    voices = {code: str(root / "voices" / f"{code}.flac") for code in codes}
    return {
        "languages": templates["languages"],
        "templates": templates["templates"],
        "presets": presets,
        "voices": {code: path for code, path in voices.items() if Path(path).is_file()},
    }


class NativeServer:
    def __init__(self, backend: LocalBackend, language: str = "ja", speak_first: bool = False, seed: int | None = None):
        self.backend = backend
        self.language = language
        self.speak_first = speak_first
        self.seed = seed
        self.assets = bundled_assets()
        if language not in self.assets["templates"]:
            raise ValueError(f"language must be one of {sorted(self.assets['templates'])}")
        self.default_card = render_card(self.assets["presets"][language][0]["card"], self.assets["templates"][language])
        voice = self.assets["voices"].get(language)
        self.default_voice = prepare_reference(load_audio(voice)) if voice else None
        self.step_seconds = 0.0
        self._busy = asyncio.Lock()

    def measure_step(self) -> None:
        """Time a frame at the live cadence (on the MLX thread, after loading).

        Audio arrives in small blocks over a frame's 133 ms and the CPU is kept
        awake before the step, as in a conversation: back to back a step is
        faster than live, and after an idle frame slower.
        """
        cfg = self.backend.model.config
        session = self.backend.session(SessionSettings(self.default_card, self.default_voice, seed=0))
        session.start()
        blocks = 20
        block = np.zeros(cfg.frame_samples // blocks, dtype=np.float32)
        period = cfg.frame_samples / cfg.sample_rate / blocks
        warm = self.backend.worker.warm
        steps = []
        for _ in range(6):
            start = time.perf_counter()
            for i in range(blocks):
                session.feed(block)
                due = start + (i + 1) * period
                if due - start < blocks * period - warm:
                    time.sleep(max(0.0, due - time.perf_counter()))
                else:
                    while time.perf_counter() < due:
                        time.sleep(0)
            steps.append(session.step().seconds)
        self.step_seconds = float(np.median(steps)) + STEP_ALLOWANCE_SECONDS

    # ------------------------------------------------------------ requests

    def _settings(self, request: dict[str, Any]) -> SessionSettings:
        """A client's START / PREPARE request -> the conversation's settings."""
        system_prompt = self.default_card
        if request.get("system_prompt"):
            system_prompt = str(request["system_prompt"])
        elif request.get("card"):
            language = request.get("language", self.language)
            system_prompt = render_card(request["card"], self.assets["templates"][language])
        reference = self.default_voice
        if request.get("reference_path"):
            reference = prepare_reference(load_audio(request["reference_path"]))
            seconds = len(reference) / self.backend.model.config.sample_rate
            if seconds < REFERENCE_MIN_SECONDS:
                raise ValueError(
                    f"the voice is too short ({seconds:.1f} s after trimming silence); "
                    f"it needs at least {REFERENCE_MIN_SECONDS:g} s of speech"
                )
        guidance = float(request.get("guidance_scale", 1.5))
        return SessionSettings(
            system_prompt=system_prompt,
            reference=reference,
            guidance_scale=min(max(guidance, 1.0), 2.0),  # the trained range
            speak_first=bool(request.get("speak_first", self.speak_first)),
            seed=self.seed,
        )

    def _prepare(self, request: dict[str, Any]):
        """``(session, model frame 0)`` with the prompt run; on the MLX thread."""
        session = self.backend.session(self._settings(request))
        return session, session.start()

    async def _wait_for_start(self, read_message, send_json):
        """Serve PREPARE messages until START; returns START's ``(session, frame 0)``."""
        worker = self.backend.worker
        ready = None  # (request key, session, frame 0)
        building = None  # (request key, task)
        seq = 0

        async def build(key: str, request: dict[str, Any], seq: int) -> None:
            nonlocal ready
            try:
                session, first = await worker.run(self._prepare, request)
            except Exception as error:
                await send_json(READY, {"seq": seq, "ok": False, "message": str(error)})
                return
            ready = (key, session, first)
            logger.info("prepared: %d prompt positions in %.2f s", session.prompt_length, first.seconds)
            await send_json(READY, {"seq": seq, "ok": True})

        try:
            while True:
                kind, payload = await read_message()
                request = json.loads(payload) if payload else {}
                key = json.dumps(request, sort_keys=True)
                if kind == START:
                    break
                if kind != PREPARE:
                    continue
                seq += 1
                if building:
                    building[1].cancel()  # skipped by the MLX thread unless it already runs
                    building = None
                if ready and ready[0] == key:
                    await send_json(READY, {"seq": seq, "ok": True})
                    continue
                ready = None
                building = (key, asyncio.ensure_future(build(key, request, seq)))
            if building and building[0] == key:
                await building[1]
            if ready and ready[0] == key:
                return ready[1:]
            return await worker.run(self._prepare, request)
        finally:
            if building:
                building[1].cancel()

    # ------------------------------------------------------------- clients

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        worker = self.backend.worker
        cfg = self.backend.model.config
        frame_samples, sample_rate = cfg.frame_samples, cfg.sample_rate
        write_lock = asyncio.Lock()

        async def send(kind: int, payload: bytes) -> None:
            async with write_lock:
                writer.write(struct.pack("<BI", kind, len(payload)) + payload)
                await writer.drain()

        async def send_json(kind: int, value: dict[str, Any]) -> None:
            await send(kind, json.dumps(value, ensure_ascii=False).encode("utf-8"))

        async def read_message() -> tuple[int, bytes]:
            kind, length = struct.unpack("<BI", await reader.readexactly(5))
            return kind, await reader.readexactly(length)

        async def send_frame(frame: Frame) -> None:
            pcm = upsample_2x(frame.audio).astype("<f4").tobytes()
            await send(FRAME, struct.pack("<If", frame.index, 1000 * frame.seconds) + pcm)
            if frame.text:
                await send(TEXT, frame.text.encode("utf-8"))

        step_ms = 1000 * self.step_seconds
        quantize = self.backend.quantize
        await send_json(
            ASSETS,
            {
                **self.assets,
                "default_language": self.language,
                "step_ms": step_ms,
                "precision": f"int{quantize}" if quantize else "bf16",
                "model": self.backend.model_path,
            },
        )
        try:
            session, first = await self._wait_for_start(read_message, send_json)
        except (asyncio.IncompleteReadError, ConnectionError):
            raise
        except Exception as error:
            await send_json(ERROR, {"message": str(error)})
            return

        ratio = CLIENT_SAMPLE_RATE // sample_rate
        await send_json(
            HELLO,
            {"sample_rate": CLIENT_SAMPLE_RATE, "frame_samples": frame_samples * ratio, "step_ms": step_ms},
        )
        await send_frame(first)
        logger.info("conversation started (%d prompt positions)", session.prompt_length)

        complete: asyncio.Queue[float] = asyncio.Queue()  # arrival time of each whole user frame
        steps: list[float] = []

        async def generate() -> None:
            while True:
                arrived = await complete.get()
                frame = await worker.run(session.step)
                worker.expect(arrived + frame_samples / sample_rate)
                steps.append(frame.seconds)
                await send_frame(frame)
                if frame.done:
                    return

        task = asyncio.ensure_future(generate())
        received = queued = 0
        leftover = np.zeros(0, dtype=np.float32)
        try:
            while True:
                kind, payload = await read_message()
                if kind != AUDIO:
                    continue
                x = np.concatenate([leftover, np.frombuffer(payload, dtype="<f4")])
                even = len(x) // 2 * 2
                leftover = x[even:]
                pcm = downsample_2x(x[:even])
                if not len(pcm):
                    continue
                worker.submit(session.feed, pcm)
                received += len(pcm)
                while received >= (queued + 1) * frame_samples:
                    queued += 1
                    complete.put_nowait(time.perf_counter())
        finally:
            task.cancel()
            worker.expect(None)
            if steps:
                ms = 1000 * np.array(steps)
                logger.info(
                    "client left: %d frames, step ms p50 %.1f p95 %.1f max %.1f (estimate %.1f)",
                    len(steps), np.percentile(ms, 50), np.percentile(ms, 95), ms.max(), step_ms,
                )

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self._busy.locked():
            writer.close()
            return
        async with self._busy:
            logger.info("client connected")
            try:
                await self._serve(reader, writer)
            except (asyncio.IncompleteReadError, ConnectionError):
                pass
            finally:
                writer.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="haru-mlx-native", description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", help="checkpoint directory")
    source.add_argument("--hf-repo", help=f"Hugging Face repository (default: {DEFAULT_REPO})")
    parser.add_argument(
        "--quantize", choices=("8", "none"), default="none",
        help="'none' keeps the checkpoint's bfloat16; '8' stores the backbone with 8 bits per weight "
        "(a frame is faster and the model smaller)",
    )
    parser.add_argument("--socket", default=default_socket(), help="default: haru.sock in the temporary directory")
    parser.add_argument("--language", default="ja", help="language of the default card and voice")
    parser.add_argument("--speak-first", action="store_true", help="the model opens the conversation")
    parser.add_argument("--seed", type=int, help="fixed sampling seed for every conversation")
    parser.add_argument("--warm-ms", type=float, default=30.0, help="keep the CPU awake this long before each frame is due (0: off)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    backend = LocalBackend(
        args.model or args.hf_repo or DEFAULT_REPO,
        quantize=None if args.quantize == "none" else int(args.quantize),
        warm_seconds=args.warm_ms / 1000,
    )
    server = NativeServer(backend, args.language, args.speak_first, args.seed)
    logger.info("loading %s", backend.model_path)
    backend.load_blocking()
    backend.worker.submit(server.measure_step).result()

    async def run() -> None:
        if os.path.exists(args.socket):
            os.unlink(args.socket)
        listener = await asyncio.start_unix_server(server.handle, path=args.socket)
        # Clients that start this process wait for this line.
        print(f"[native] ready on {args.socket} (step {1000 * server.step_seconds:.0f} ms)", flush=True)
        async with listener:
            await listener.serve_forever()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
