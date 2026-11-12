# Haru for macOS

`Haru.app` lets you talk with Haru on your Mac: it listens and speaks at the
same time, with backchannels and interruptions. The app does the audio with
Core Audio and runs the model in a Python process (`haru_mlx`, on MLX) that it
starts by itself.

## Requirements

- A Mac with Apple silicon, macOS 13 or later. The model is about 10 GB on
  disk; 16 GB of memory is the practical minimum.
- The Command Line Tools (for `swiftc`; Xcode itself is not needed).
- Python 3.10 or later.
- Wired headphones and a wired or USB microphone (see [Audio devices](#audio-devices)).

## Setup

Run these from the `haru_mlx` directory (the parent of this one).

**1. Install the Command Line Tools** (skip if `xcode-select -p` prints a path):

```bash
xcode-select --install
```

**2. Create the Python environment** in `.venv` (the build looks for it there):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

**3. Build the app:**

```bash
cd swift
./build_app.sh
```

This writes `build/Haru.app` and the command-line client `.build/haru-client`.
To use another Python, build with `PYTHON=/path/to/python ./build_app.sh`.

**4. Open it:**

```bash
open build/Haru.app
```

Allow microphone access when asked. The app starts the model server
(`python -m haru_mlx.native`); the first launch downloads the model
(`higurashi-ai/Haru-4B`) from Hugging Face.

Build the app on each Mac: it records the path of its Python when it is built,
so a copy moved to another Mac cannot start the server.

## Using the app

1. **Character** (キャラクター): pick an example or write the card: the
   model's name, your name, the model's gender and backchannels, its persona,
   your relationship and the situation.
2. **Voice** (声): an audio file or a recording of 3 seconds to 3 minutes of
   one speaker. Haru ships without voice clips; bring your own.
3. **Settings** (設定): the microphone and the output, the guidance scale, the
   margin (playback slack for a late model step) and the precision: BF16
   (the default, as published) or int8 (smaller and faster; switching reloads
   the model).
4. Click **会話を始める** and talk. A few seconds after your last edit of the
   card or the voice, the server reads them into the model in the background,
   and **準備完了** appears next to the button; the conversation then starts as
   soon as you click. The screen shows your voice and the model's as two
   waveform lanes on one clock, the model's words as they are heard, and the
   reply latency.
5. Click **会話を終える**. You can then save the conversation's audio (WAV: you
   on the left, the model on the right, as heard) or a video of the screen's
   lanes and transcript (MP4).

Quitting the app stops the server. The server's output and each
conversation's latency are logged to `~/Library/Logs/Haru/app.log`.

## Audio devices

Use wired headphones and a wired or USB microphone. The devices' own latency
adds to the reply latency, and with speakers the model hears itself.
Bluetooth headphones add 100 ms and more; the built-in microphone and speakers
of a MacBook add tens of milliseconds of processing. The app picks a wired or
USB microphone first and warns when the chosen devices add a lot of latency.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `swiftc not found` when building | Install the Command Line Tools (step 1). |
| `haru_mlx is not installed` when building | Create the environment (step 2), or pass `PYTHON=`. |
| The app says the server could not start | The launch screen shows the server's output. Try `python -m haru_mlx.native` in a terminal. |
| No sound | Check the output in Settings: the app uses the device chosen there. |
| Choppy playback (途切れ) | Raise the margin, use wired devices, and close heavy apps. |
| "Unidentified developer" warning | Build the app on that Mac instead of copying it. |

## Command-line client

`.build/haru-client` talks to a server you start yourself, with the server's
default card; on exit it prints the reply latency.

```bash
python -m haru_mlx.native            # in one terminal; wait for "ready on ..."
.build/haru-client --list-devices
.build/haru-client --input "MacBook Pro Microphone" --output "MacBook Pro Speakers"
```

Options: `--input`, `--output` (device names, default: the system devices),
`--buffer-ms` (2), `--margin-ms` (15), `--seconds`, `--socket` (default:
`haru.sock` in `$TMPDIR`, the server's default).

## How it works

The app and the client do the audio I/O with Core Audio (HAL output units with
small device buffers) and talk to the model server over a Unix socket; the
messages are described at the top of `haru_mlx/native.py`. Input devices that
do not run at 48 kHz are resampled in the client; output runs at 48 kHz.

Both clients measure the reply latency on the audio clock, from the last
sample of each user frame at the microphone converter to the first sample of
the model frame that answers it at the speaker converter. Model frame 0 starts
playing when model frame 1 can follow it without a gap: after the input
buffer, the model step (measured by the server at start-up), the output buffer
and device latency, and the margin. When a frame arrives with more than the
margin queued, the client shortens it slightly (pitch unchanged) until
playback is back on schedule, so a late step costs a moment of silence instead
of permanent latency.
