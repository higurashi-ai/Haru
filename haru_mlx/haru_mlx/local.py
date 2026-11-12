"""Talk to Haru in a terminal, with this Mac's microphone and speakers.

    python -m haru_mlx.local --reference voice.wav --system-prompt "..."

The model's text streams to the terminal while you talk. Use headphones: the
model would otherwise hear its own voice. Without ``--reference`` or a system
prompt, the bundled example card and voice of ``--language`` are used.

Audio goes straight from the input device into the model and back out through
a small queue. At the end the reply latency is printed, measured on the audio
clock: from the last sample of a user frame entering the microphone converter
to the first sample of the model frame that answers it leaving the speaker
converter.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import deque
from importlib import resources
from pathlib import Path

import numpy as np

from .audio import load_audio, prepare_reference, save_audio
from .backends import LocalBackend, SessionSettings
from .hub import DEFAULT_REPO
from .prompt import card_templates, render_card


class Player:
    """Output queue filled by the model thread and drained by the audio callback.

    The reply to user frame ``t`` is model frame ``t + 1``, which exists one
    audio block plus one model step after user frame ``t`` is complete. Model
    frame 0 (from the prompt) therefore waits ``lead`` seconds after the stream
    starts, so that frame 1 is ready when frame 0 ends. Whenever the queue runs
    dry, ``margin`` seconds of silence are played before audio resumes, which
    restores the slack for a late step.
    """

    def __init__(self, sample_rate: int, margin: float, lead: float):
        self.sample_rate = sample_rate
        self.underruns = 0
        self.played_at: dict[int, float] = {}  # frame index -> time its first sample reaches the speaker
        self._margin = int(margin * sample_rate)
        self._silence = int(lead * sample_rate)  # samples of silence still to play
        self._frames: deque[tuple[int, np.ndarray]] = deque()
        self._offset = 0
        self._dry = False
        self._lock = threading.Lock()

    def push(self, index: int, audio: np.ndarray) -> None:
        with self._lock:
            self._frames.append((index, audio))

    def pull(self, out: np.ndarray, speaker_time: float) -> None:
        """Fill ``out``; ``speaker_time`` is when its first sample will be played."""
        n, written = len(out), 0
        with self._lock:
            if self._dry and self._frames:
                self._dry = False
                self._silence = self._margin
            while written < n:
                if self._silence:
                    take = min(self._silence, n - written)
                    out[written : written + take] = 0
                    written += take
                    self._silence -= take
                    continue
                if not self._frames:
                    break
                index, audio = self._frames[0]
                if self._offset == 0:
                    self.played_at[index] = speaker_time + written / self.sample_rate
                take = min(len(audio) - self._offset, n - written)
                out[written : written + take] = audio[self._offset : self._offset + take]
                written += take
                self._offset += take
                if self._offset == len(audio):
                    self._frames.popleft()
                    self._offset = 0
            if written < n and not self._dry:
                self._dry = True
                self.underruns += 1
        out[written:] = 0


def example_prompt(language: str) -> tuple[str, Path]:
    """The bundled example card (rendered) and voice file of ``language``."""
    assets = resources.files("haru_mlx") / "assets"
    templates = card_templates()["templates"]
    if language not in templates:
        raise SystemExit(f"--language must be one of {sorted(templates)}")
    cards = json.loads((assets / "cards" / f"{language}.json").read_text(encoding="utf-8"))
    return render_card(next(iter(cards.values())), templates[language]), Path(str(assets / "voices" / f"{language}.flac"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="haru-mlx-local", description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", help="checkpoint directory")
    source.add_argument("--hf-repo", help=f"Hugging Face repository (default: {DEFAULT_REPO})")
    parser.add_argument(
        "--quantize", choices=("8", "none"), default="8",
        help="'8' stores the backbone with 8 bits per weight (a frame is faster and the model smaller); "
        "'none' keeps the checkpoint's bfloat16",
    )
    parser.add_argument("--language", default="ja", help="language of the bundled example card and voice")
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--system-prompt", help="character card of the model, as text")
    prompt.add_argument("--system-prompt-file", type=Path, help="read the system prompt from a file")
    parser.add_argument("--reference", type=Path, help="voice the model speaks with (3-180 s of one speaker)")
    parser.add_argument("--guidance-scale", type=float, default=1.5, help="audio guidance scale (1-2)")
    parser.add_argument("--temperature", type=float, default=0.7, help="text sampling temperature")
    parser.add_argument("--top-k", type=int, default=25, help="text sampling top-k")
    parser.add_argument("--seed", type=int, help="sampling seed")
    parser.add_argument("--speak-first", action="store_true", help="the model opens the conversation")
    parser.add_argument("--input-device", help="name or index (see --list-devices)")
    parser.add_argument("--output-device", help="name or index (see --list-devices)")
    parser.add_argument("--list-devices", action="store_true", help="print the audio devices and exit")
    parser.add_argument("--block-ms", type=float, default=5.0, help="audio callback period")
    parser.add_argument("--margin-ms", type=float, default=20.0, help="playback slack for a late model step")
    parser.add_argument("--warm-ms", type=float, default=30.0, help="keep the CPU awake this long before each frame is due (0: off)")
    parser.add_argument("--seconds", type=float, help="stop after this long")
    parser.add_argument("--save", type=Path, help="write the conversation as a stereo WAV (left: you, right: the model)")
    return parser


def _device(value: str | None) -> int | str | None:
    return int(value) if value is not None and value.isdigit() else value


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit('the terminal client needs sounddevice: pip install "haru_mlx[local]"') from None
    if args.list_devices:
        print(sd.query_devices())
        return

    system_prompt = args.system_prompt
    if args.system_prompt_file:
        system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip()
    reference_path = args.reference
    if system_prompt is None or reference_path is None:
        card, voice = example_prompt(args.language)
        system_prompt = card if system_prompt is None else system_prompt
        reference_path = voice if reference_path is None else reference_path
    if not Path(reference_path).is_file():
        raise SystemExit(f"no voice file at {reference_path}: pass --reference")
    reference = prepare_reference(load_audio(reference_path))

    print("loading the model...", file=sys.stderr)
    backend = LocalBackend(
        args.model or args.hf_repo or DEFAULT_REPO,
        quantize=None if args.quantize == "none" else int(args.quantize),
        warm_seconds=args.warm_ms / 1000,
    )
    backend.load_blocking()
    worker = backend.worker
    cfg = backend.model.config
    sample_rate, frame_samples = cfg.sample_rate, cfg.frame_samples
    settings = SessionSettings(
        system_prompt=system_prompt,
        reference=reference,
        guidance_scale=args.guidance_scale,
        temperature=args.temperature,
        top_k=args.top_k,
        speak_first=args.speak_first,
        seed=args.seed,
    )
    session = worker.submit(backend.session, settings).result()
    first = worker.submit(session.start).result()

    # Time a few steps on silence to know how far playback must trail the microphone.
    probe = worker.submit(backend.session, settings).result()
    worker.submit(probe.start).result()
    silence = np.zeros(frame_samples, dtype=np.float32)
    step_seconds = float(np.median([worker.submit(probe.step, silence).result().seconds for _ in range(8)]))
    del probe

    margin = args.margin_ms / 1000
    player = Player(sample_rate, margin=margin, lead=args.block_ms / 1000 + step_seconds + margin)
    player.push(0, first.audio)

    recorded_at: dict[int, float] = {}  # user frame -> time its last sample entered the microphone
    compute: list[float] = []
    user_audio: list[np.ndarray] = []
    model_audio: list[np.ndarray] = [first.audio]
    received = queued = 0
    done = threading.Event()

    def step(index: int, arrived: float) -> None:
        frame = session.step()
        compute.append(frame.seconds)
        player.push(index + 1, frame.audio)
        worker.expect(arrived + frame_samples / sample_rate)
        model_audio.append(frame.audio)
        if frame.text:
            sys.stdout.write(frame.text)
            sys.stdout.flush()
        if frame.done:
            done.set()

    def report(future) -> None:
        if not future.cancelled() and future.exception() is not None:
            print(f"\n[error] {future.exception()!r}", file=sys.stderr)
            done.set()

    def callback(indata, outdata, frames, timeinfo, status) -> None:
        nonlocal received, queued
        microphone = indata[:, 0].copy()
        start = received
        received += frames
        worker.submit(session.feed, microphone).add_done_callback(report)
        if args.save:
            user_audio.append(microphone)
        while received >= (queued + 1) * frame_samples:
            last = (queued + 1) * frame_samples - 1 - start  # the frame's last sample within this block
            recorded_at[queued] = timeinfo.inputBufferAdcTime + last / sample_rate
            worker.submit(step, queued, time.perf_counter()).add_done_callback(report)
            queued += 1
        player.pull(outdata[:, 0], timeinfo.outputBufferDacTime)

    stream = sd.Stream(
        samplerate=sample_rate,
        blocksize=int(sample_rate * args.block_ms / 1000),
        channels=1,
        dtype="float32",
        latency="low",
        device=(_device(args.input_device), _device(args.output_device)),
        callback=callback,
    )
    print(
        f"input: {sd.query_devices(stream.device[0])['name']} | output: {sd.query_devices(stream.device[1])['name']} | "
        f"model step {1000 * step_seconds:.0f} ms per {1000 * frame_samples / sample_rate:.0f} ms frame. "
        "Talk now (Ctrl+C to stop).\n",
        file=sys.stderr,
    )
    started = time.monotonic()
    try:
        with stream:
            while not done.wait(0.2):
                if args.seconds and time.monotonic() - started > args.seconds:
                    break
    except KeyboardInterrupt:
        pass
    worker.expect(None)
    worker.submit(lambda: None).result()  # let the queued steps finish

    latencies = [
        1000 * (played - recorded_at[index - 1])
        for index, played in sorted(player.played_at.items())
        if index - 1 in recorded_at
    ]
    print(file=sys.stderr)
    if latencies:
        print(
            f"[latency] {len(latencies)} frames: end of a user frame (microphone) -> reply (speaker) "
            f"p50 {np.percentile(latencies, 50):.0f} ms, p95 {np.percentile(latencies, 95):.0f} ms | "
            f"model step p50 {1000 * np.median(compute):.1f} ms | underruns {player.underruns}",
            file=sys.stderr,
        )
    if args.save:
        user = np.concatenate(user_audio) if user_audio else np.zeros(0, dtype=np.float32)
        reply = np.concatenate(model_audio)
        n = min(len(user), len(reply))
        save_audio(args.save, np.stack([user[:n], reply[:n]], axis=1))
        print(f"wrote {args.save}", file=sys.stderr)


if __name__ == "__main__":
    main()
