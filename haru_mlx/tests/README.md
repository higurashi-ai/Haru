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

Run on Linux x86-64 with the CPU build of MLX 0.32 (`pip install "mlx[cpu]"`),
which takes the array fallbacks instead of the Metal kernels:

- all tests above, with the tiny checkpoint and with `Haru-4B`;
- parity in float32 against PyTorch on `Haru-4B`: hidden states 2.5e-3 over
  the prompt and 2e-4 on single steps (relative error), identical top-50 text
  tokens, head 2e-6, codec 2e-6 (encoder) and 2e-5 (decoder);
- parity in bfloat16: hidden states 6e-2, which is below the 9e-2 that
  separates PyTorch's own bfloat16 and float32 results;
- the 8-bit backbone against float32 PyTorch: hidden states 9e-2 and the same
  text tokens, the level of the bfloat16 checkpoint (4-bit weights were tried
  and rejected: hidden states off by about 100 %);
- the commands end to end with the tiny checkpoint: `run_inference` and the
  native socket server.

Not yet run on a Mac. Checklist:

1. `pytest` and `HARU_MODEL=... pytest` pass (this runs the Metal kernels of
   `models/gated_delta.py` and `models/vibevoice_codec.py` for the first time).
2. `test_parity.py` passes in bfloat16 and in float32 with dumps made as above.
3. Kernels against their fallbacks: run `test_parity.py` once more with
   `mx.set_default_device(mx.cpu)` at the top of `conftest.py`; both devices
   must pass against the same dump.
4. Real-time factor: `python -m haru_mlx.run_inference user.wav reply.wav
   --reference voice.wav` prints the median time per frame; it must stay well
   below 133 ms for each `--quantize` setting (`none`, `8`), and the
   reply must sound right.
5. Memory: peak while loading and resident while talking, for each
   `--quantize` setting, on a 16 GB machine.
6. `swift/build_app.sh`, then `swift/build/Haru.app`: the app starts
   `python -m haru_mlx.native`, the example cards and voices load, a
   conversation runs, the precision switch reloads the model; and
   `swift/.build/haru-client` against a server started by hand.
