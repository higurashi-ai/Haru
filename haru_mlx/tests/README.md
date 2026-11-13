# Tests

```bash
pytest                                   # no checkpoint needed
HARU_MODEL=/path/to/Haru-4B pytest       # + checkpoint coverage and a short conversation
```

| File | Needs |
|---|---|
| `test_sampling.py`, `test_prompt.py`, `test_audio.py`, `test_weights.py`, `test_imports.py` | nothing (two tests in sampling / prompt need MLX) |
| `test_shared_code.py` | the PyTorch package `haru` importable (its source is compared, not run) |
| `test_model.py` | MLX; the conversation tests also need `HARU_MODEL` |
| `test_parity.py` | MLX, `HARU_MODEL`, `HARU_REFERENCE` |

## Numerical parity with the PyTorch package

`parity/dump_reference.py` runs `haru` on fixed inputs and saves inputs and
outputs; `test_parity.py` feeds the same inputs to `haru_mlx`.

```bash
# where PyTorch runs
python tests/parity/dump_reference.py --model Haru-4B --out reference.npz                    # bfloat16
python tests/parity/dump_reference.py --model Haru-4B --out reference32.npz --dtype float32

# where MLX runs
HARU_MODEL=Haru-4B HARU_REFERENCE=reference.npz pytest tests/test_parity.py
HARU_MODEL=Haru-4B HARU_REFERENCE=reference32.npz HARU_PARITY_DTYPE=float32 pytest tests/test_parity.py
```

`parity/make_tiny_checkpoint.py` writes a small random checkpoint with every
layer type, for the same comparison in seconds.

## Status of the checks

Tested on a MacBook Pro with an M1 Max chip (64 GB, macOS 26.6, MLX 0.32.4),
with `Haru-4B`; the PyTorch dumps were made on the same machine's CPU:

- all tests above pass, with the Metal kernels of `models/gated_delta.py` and
  `models/vibevoice_codec.py`;
- parity in float32 against PyTorch: hidden states 9e-6 over the prompt and
  6e-6 on single steps (relative error), the same top text token, head 2e-6,
  codec 2e-6 (encoder) and 2e-5 (decoder);
- the same with MLX on the CPU (`mx.set_default_device(mx.cpu)`), which takes
  the array fallbacks instead of the kernels: 1e-5 or better throughout;
- parity in bfloat16: hidden states 1.1e-1 to 1.4e-1 on the GPU and 6e-2 to
  9e-2 on the CPU, where PyTorch's own bfloat16 and float32 results are 6e-2
  to 7e-2 apart; decoded audio 2e-1 to 3e-1, as between PyTorch's bfloat16
  and float32;
- real time: `python -m haru_mlx.run_inference` on a 31 s recording takes
  43 ms per frame in bfloat16 and 31 ms with `--quantize 8` (a frame is
  133 ms of audio), with a peak memory footprint of 13.5 GB and 11.2 GB;
- `swift/build_app.sh` builds `Haru.app`, which starts the model server and
  holds a live conversation.
