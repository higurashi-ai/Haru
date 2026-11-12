# haru_mlx

Haru on Apple silicon: the [MLX](https://github.com/ml-explore/mlx)
implementation of the Haru full-duplex speech language model. It loads the
same checkpoint as the PyTorch package `haru` (`higurashi-ai/Haru-4B`) and has
the same Python API.

Haru listens and speaks at the same time. Every 133 ms it takes one frame of
your audio and produces one frame of its own speech together with its text.

## Install

```bash
pip install -e .                 # from this directory
pip install -e ".[local]"        # + microphone / speaker client
pip install -e ".[webrtc]"       # + WebRTC transport of the web demo
```

Requires a Mac with Apple silicon and Python 3.10 or later. The checkpoint is
about 10 GB; it is downloaded from Hugging Face on first use.

## Talk to it

```bash
python -m haru_mlx.local --language en --reference voice.wav   # terminal: your microphone and speakers
python -m haru_mlx.server                         # web demo on http://localhost:8998
python -m haru_mlx.run_inference user.wav reply.wav --reference voice.wav --system-prompt "..."
```

Haru ships without voice clips: `voice.wav` is a clip of the voice the model
should speak with (one speaker, 3 seconds to 3 minutes), and the web demo asks
you to upload or record one. Only use voices you have the right to use.

Use headphones with `haru_mlx.local`, or the model hears itself. Every command
takes `--model DIR` or `--hf-repo REPO` for the checkpoint, and `--quantize`:

| `--quantize` | Backbone weights |
|---|---|
| `8` (default of the live commands) | 8-bit: smaller, and a frame reads fewer bytes |
| `none` (default of `run_inference`) | bfloat16, as published |

Quantization covers the language model's weight matrices and the embedding
table; the audio codec, the audio head and all normalization gains keep the
checkpoint's precision.

A native macOS app is in [`swift/`](swift/README.md).

## Python API

```python
import numpy as np
from haru_mlx import DuplexConfig, DuplexSession, HaruModel
from haru_mlx.audio import load_audio, prepare_reference

model = HaruModel.from_pretrained("higurashi-ai/Haru-4B", quantize=8)
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
from one thread (`haru_mlx.worker.MLXWorker` is that thread for
the server and the clients).

## Layout

| Module | |
|---|---|
| `models/haru.py` | `HaruModel`: the three input streams, the backbone, text logits |
| `models/qwen3_5.py` | Qwen3.5 text backbone (Gated DeltaNet + attention), laid out for one position per frame |
| `models/gated_delta.py` | the Gated DeltaNet recurrence (Metal kernel, with an array fallback) |
| `models/meanflow_head.py` | one-step MeanFlow head: hidden state -> audio latent |
| `models/vibevoice_codec.py` | VibeVoice acoustic codec, streaming |
| `models/loaders.py`, `weights.py` | checkpoint loading; the tensor -> parameter mapping |
| `session.py` | `DuplexSession`: one conversation |
| `local.py`, `server.py`, `native.py`, `run_inference.py` | the commands |

## Tests

```bash
pip install -e ".[dev]"
pytest                                   # the tests that need no checkpoint
HARU_MODEL=/path/to/Haru-4B pytest       # + checkpoint coverage and a short conversation
```

Most tests run without MLX. [`tests/README.md`](tests/README.md) describes the
numerical comparison with the PyTorch package and the checks to run on a Mac.
