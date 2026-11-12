# haru_vllm

Serve Haru with [vLLM-Omni](https://github.com/vllm-project/vllm-omni): many
concurrent full-duplex conversations on one GPU, over the realtime WebSocket
API.

This is a plugin. Installing it registers the Haru model, pipeline and duplex
policy with vLLM-Omni; nothing in vLLM-Omni is patched.

## Install and run

```bash
pip install vllm==0.33.0 vllm-omni      # see the vLLM-Omni installation guide
pip install -e ../haru -e .             # from this directory

vllm serve higurashi-ai/Haru-4B --omni --deploy-config "$(python -m haru_vllm.deploy)"
```

The server listens on `ws://localhost:8000/v1/realtime?duplex=1`. To use it
from the web demo:

```bash
python -m haru.server --backend vllm --vllm-url http://localhost:8000
```

`python -m haru_vllm.deploy` prints the path of the bundled deploy
configuration ([`deploy/haru.yaml`](haru_vllm/deploy/haru.yaml)). Copy and edit
it to change the number of concurrent conversations (`duplex_session.max_sessions`
and `max_num_seqs`) or the cache size.

Tested with vLLM 0.33.0 and vLLM-Omni at commit `04e03a4`. The plugin uses
model-runner hooks of vLLM-Omni that may change between versions.

## Protocol

The endpoint speaks vLLM-Omni's realtime duplex protocol (OpenAI Realtime
events). One conversation:

1. `session.update` opens the session. `instructions` is the character card,
   `ref_audio` a `data:audio/...;base64,` URI of the voice. Audio input is
   `pcm_f32le` at 24 kHz.
2. Append one frame (3200 samples) with `input_audio_buffer.append`. This first
   append starts the conversation: the prompt runs and the model's frame 0
   comes back. Its audio is not used; send silence.
3. From then on, every appended frame `k` is answered by model frame `k + 1`:
   one `response.output_audio.delta` (PCM16, 24 kHz, 3200 samples) and, when
   the frame has text, a `response.output_audio_transcript.delta`.

Options in `session.extra_body`, fixed when the session is created:

| Key | Default | Meaning |
| --- | --- | --- |
| `guidance_scale` | 1.5 | Audio guidance, 1 to 2 |
| `top_k`, `top_p` | 25, 1.0 | Text sampling (with the session's `temperature`, default 0.7) |
| `eop_bias` | 0 | Added to the logit of the "start speaking" token |
| `speak_first` | false | The model opens the conversation |
| `dry` | true | Repetition penalty on the text stream |
| `seed` | random | Seed of the audio sampling |

## How it is built

One autoregressive stage. Each conversation is a single resumable engine
request whose prompt grows by one position per frame; the language model runs
on vLLM's engine (paged attention cache, Gated DeltaNet state, CUDA graphs).
The audio parts run in model hooks of the same step: the user frame is encoded
before the forward pass, and the MeanFlow head and codec decoder run right
after sampling, so a frame's audio leaves in the step that produced it.
