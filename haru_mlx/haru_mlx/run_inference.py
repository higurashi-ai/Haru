"""Offline inference: play a recording of the user to Haru and save what it answers.

    python -m haru_mlx.run_inference user.wav reply.wav --system-prompt "..." --reference voice.wav

The recording is fed frame by frame, exactly as in a live conversation; the
output has the same length. ``--transcript`` writes one JSON line per frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE, load_audio, prepare_reference, save_audio
from .hub import DEFAULT_REPO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="haru-mlx-inference", description=__doc__.split("\n\n")[0])
    parser.add_argument("input", type=Path, help="audio of the user (any rate; mixed down to mono)")
    parser.add_argument("output", type=Path, help="audio of the model (24 kHz WAV)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", help="checkpoint directory")
    source.add_argument("--hf-repo", help=f"Hugging Face repository (default: {DEFAULT_REPO})")
    parser.add_argument(
        "--quantize", choices=("8", "none"), default="none",
        help="'8' stores the backbone with 8 bits per weight (default: the checkpoint's bfloat16)",
    )
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--system-prompt", default="", help="character card of the model, as text")
    prompt.add_argument("--system-prompt-file", type=Path, help="read the system prompt from a file")
    parser.add_argument("--reference", type=Path, help="voice the model speaks with (3-180 s of one speaker)")
    parser.add_argument("--guidance-scale", type=float, default=1.5, help="audio guidance scale (1-2)")
    parser.add_argument("--temperature", type=float, default=0.7, help="text sampling temperature")
    parser.add_argument("--top-k", type=int, default=25, help="text sampling top-k")
    parser.add_argument("--seed", type=int, help="sampling seed")
    parser.add_argument("--speak-first", action="store_true", help="the model opens the conversation")
    parser.add_argument("--stereo", action="store_true", help="write the user on the left channel and the model on the right")
    parser.add_argument("--transcript", type=Path, help="write per-frame text as JSON lines")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from .models import load_model
    from .session import DuplexConfig, DuplexSession

    model = load_model(
        args.model or args.hf_repo or DEFAULT_REPO, quantize=None if args.quantize == "none" else int(args.quantize)
    )
    frame_samples = model.config.frame_samples
    user = load_audio(args.input)
    count = max(1, -(-len(user) // frame_samples))
    user = np.pad(user, (0, count * frame_samples - len(user)))
    system_prompt = args.system_prompt
    if args.system_prompt_file:
        system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip()
    reference = prepare_reference(load_audio(args.reference)) if args.reference else None
    config = DuplexConfig(
        guidance_scale=args.guidance_scale,
        temperature=args.temperature,
        top_k=args.top_k,
        speak_first=args.speak_first,
        max_frames=count + 1,
        seed=args.seed,
    )
    session = DuplexSession(model, system_prompt, reference, config)

    # Model frame t is heard while the user speaks frame t, and was computed from the
    # user's frames before t: the last user frame needs no step.
    frames = [session.start()]
    for index in range(count - 1):
        frames.append(session.step(user[index * frame_samples : (index + 1) * frame_samples]))
    reply = np.concatenate([frame.audio for frame in frames])

    save_audio(args.output, np.stack([user, reply], axis=1) if args.stereo else reply)
    text = "".join(frame.text for frame in frames).strip()
    if args.transcript:
        with args.transcript.open("w", encoding="utf-8") as file:
            for frame in frames:
                record = {"frame": frame.index, "time": round(frame.index * frame_samples / SAMPLE_RATE, 3), "text": frame.text}
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
    step_ms = 1000 * float(np.median([frame.seconds for frame in frames[1:]])) if count > 1 else 0.0
    print(f"{count} frames ({count * frame_samples / SAMPLE_RATE:.1f} s), {step_ms:.1f} ms per frame -> {args.output}")
    print(text)


if __name__ == "__main__":
    main()
