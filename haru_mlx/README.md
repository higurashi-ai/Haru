# haru_mlx

<!-- This README is also the project description on PyPI, where relative links do not resolve. Link to other files in the repository with absolute GitHub URLs. -->

Haru on Apple silicon: the [MLX](https://github.com/ml-explore/mlx)
implementation of the Haru full-duplex speech language model. It loads the
same checkpoint as the PyTorch package `haru` (`higurashi-ai/Haru-4B`) and has
the same Python API.

Haru listens and speaks at the same time. Every 133 ms it takes one frame of
your audio and produces one frame of its own speech together with its text.

## Install

```bash
pip install -e .                 # from this directory
```

Requires a Mac with Apple silicon and Python 3.10 or later. The checkpoint is
about 10 GB; it is downloaded from Hugging Face on first use.

## Talk to it

**[Haru for macOS](https://github.com/higurashi-ai/Haru/blob/main/haru_mlx/swift/README.md)** is the way to talk to the model on a
Mac: a native app that handles the microphone and speakers with Core Audio and
runs this package as its model server. Build it with `swift/build_app.sh`.

Haru ships without voice clips: in the app, choose or record a clip of the
voice the model should speak with (one speaker, 3 seconds to 3 minutes). Only
use voices you have the right to use.

## Offline

```bash
python -m haru_mlx.run_inference user.wav reply.wav --reference voice.wav --system-prompt "..."
```

A recording in, the model's side of the conversation out. `--model DIR` or
`--hf-repo REPO` selects the checkpoint, and `--quantize` its precision. The
app starts in bfloat16 and can switch to 8-bit in its settings:

| `--quantize` | Backbone weights |
|---|---|
| `none` (default) | bfloat16, as published |
| `8` | 8-bit: smaller, and a frame reads fewer bytes |

Quantization covers the language model's weight matrices and the embedding
table; the audio codec, the audio head and all normalization gains keep the
checkpoint's precision.

## Python API

```python
import numpy as np
from haru_mlx import DuplexConfig, DuplexSession, HaruModel
from haru_mlx.audio import load_audio, prepare_reference

model = HaruModel.from_pretrained("higurashi-ai/Haru-4B")       # quantize=8 for an 8-bit backbone
voice = prepare_reference(load_audio("voice.wav"))          # mono float32, 24 kHz
session = DuplexSession(model, system_prompt="...", reference=voice, config=DuplexConfig(seed=0))

frame = session.start()                                     # model frame 0
while not frame.done:
    user = np.zeros(3200, dtype=np.float32)                 # 133 ms of your audio
    frame = session.step(user)                              # -> frame.audio (3200 samples), frame.text
```

`DuplexConfig` has the same fields as in `haru`. A live client can also hand
over microphone audio as it arrives, which spreads the audio encoder's work
over the frame:

```python
session.feed(block)              # any length, as the microphone delivers it
if session.frames_ready:
    frame = session.step()
```

MLX keeps its GPU streams per thread: load the model and drive its sessions
from one thread (`haru_mlx.worker.MLXWorker` is that thread for the app's
model server).

## Metal kernels

Two inner loops of a frame run as custom Metal kernels: the Gated DeltaNet
recurrence of the language model and the depthwise convolution blocks of the
audio codec. Both have plain MLX fallbacks with the same results, which are
used on devices without Metal.

## Layout

| Module | |
|---|---|
| `models/haru.py` | `HaruModel`: the three input streams, the backbone, text logits |
| `models/qwen3_5.py` | Qwen3.5 text backbone (Gated DeltaNet + attention), laid out for one position per frame |
| `models/gated_delta.py` | the Gated DeltaNet recurrence (Metal kernel, with an array fallback) |
| `models/meanflow_head.py` | one-step MeanFlow head: hidden state -> audio latent |
| `models/vibevoice_codec.py` | VibeVoice acoustic codec, streaming (Metal kernel for its depthwise convolutions, with an array fallback) |
| `models/loaders.py`, `weights.py` | checkpoint loading; the tensor -> parameter mapping |
| `session.py` | `DuplexSession`: one conversation |
| `native.py` | the model server of the macOS app (a Unix socket) |
| `run_inference.py` | offline inference |

## Tests

```bash
pip install -e ".[dev]"
pytest                                   # the tests that need no checkpoint
HARU_MODEL=/path/to/Haru-4B pytest       # + checkpoint coverage and a short conversation
```

Most tests run without MLX. [`tests/README.md`](https://github.com/higurashi-ai/Haru/blob/main/haru_mlx/tests/README.md) describes the
numerical comparison with the PyTorch package and the checks to run on a Mac.
