"""Write a small random Haru checkpoint (needs the PyTorch ``haru`` package).

    python make_tiny_checkpoint.py --tokenizer Haru-4B/tokenizer.json --out tiny-haru

The checkpoint has the layout and every layer type of the released model at a
size that loads in a second, for parity tests of ``haru_mlx`` against ``haru``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from haru.config import HaruConfig
from haru.models.haru import HaruModel

CONFIG = json.loads((Path(__file__).parent / "tiny_config.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokenizer", required=True, help="tokenizer.json of a Haru checkpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = HaruModel(HaruConfig.from_dict(CONFIG))
    state = model.state_dict()
    for name, value in state.items():
        # Every tensor random, so that no layer is an identity or a zero.
        scale = 0.02 if name == "lm.embed_tokens.weight" else 0.1 if value.dim() <= 1 else value.shape[-1] ** -0.5
        state[name] = torch.randn(value.shape) * scale
        if name.endswith(("gamma", ".norm.weight", "ffn_norm.weight")):
            state[name] += 1.0
    state["latent_scale"] = torch.tensor(0.2)
    state["latent_bias"] = torch.tensor(0.05)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(state, str(out / "model.safetensors"), metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(CONFIG, indent=2))
    shutil.copyfile(args.tokenizer, out / "tokenizer.json")
    print(f"wrote {out} ({len(state)} tensors)")


if __name__ == "__main__":
    main()
