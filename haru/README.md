# haru

The PyTorch implementation of Haru, a full-duplex speech language model, with
a web demo server and an offline inference command.

```bash
pip install -e .                  # from this directory
pip install -e ".[webrtc]"        # + WebRTC transport for the web demo
pip install -e ".[triton]"        # + faster audio codec on NVIDIA GPUs
pip install -e ".[share]"         # + public link for the web demo (--share)
```

```bash
haru-server --hf-repo higurashi-ai/Haru-4B       # web demo on http://localhost:8998
haru-inference user.wav reply.wav --reference voice.wav --system-prompt "..."
```

Requirements: Python 3.10 or later and a CUDA GPU with about 12 GB of memory.
`ffmpeg` on the `PATH` is needed to use voice clips recorded in the browser
(WebM / MP4); WAV, FLAC and Ogg files work without it.

## Layout

| Module | Contents |
| --- | --- |
| `haru.models.haru` | `HaruModel`: language model, audio projections, audio head, codec |
| `haru.models.meanflow_head` | `MeanFlowHead`: one audio latent from one hidden state |
| `haru.models.vibevoice_codec` | `VibeVoiceCodec`: streaming audio encoder and decoder |
| `haru.codec_triton` | The same codec as Triton kernels replayed from CUDA graphs |
| `haru.session` | `DuplexSession`: a live conversation, one frame at a time |
| `haru.prompt` | Character cards and the prompt layout |
| `haru.sampling` | Text sampling, repetition penalty, incremental detokenization |
| `haru.server`, `haru.webrtc`, `haru.backends` | The web demo |
| `haru.run_inference` | Offline: a recording in, the model's side of the conversation out |

See the [repository README](../README.md) for the Python API.

## Tests

```bash
pip install -e ".[dev]"
pytest                                  # CPU-only tests
HARU_MODEL=/path/to/Haru-4B pytest      # + tests that load the checkpoint on a GPU
```
