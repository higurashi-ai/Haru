"""Run the PyTorch ``haru`` package on fixed inputs and save inputs and outputs.

    python dump_reference.py --model Haru-4B --out reference.npz [--dtype float32]

``tests/test_parity.py`` feeds the saved inputs to ``haru_mlx`` and compares
(set ``HARU_MODEL`` and ``HARU_REFERENCE``). Nothing is sampled: token ids,
latents, audio and noise are all fixed here.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from haru.models import CodecStream, load_model

PROMPT = "You are Haru."
PREFILL_AUDIO = 24  # reference positions after the system prompt
STEPS = 6  # single positions after the prefill
CLIP_SAMPLES = 5 * 3200 + 1234  # a clip that ends inside a frame


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # Exact float32 arithmetic: TF32 convolutions alone cost three digits.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_model(args.model, device=args.device, dtype=getattr(torch, args.dtype) if args.dtype else None)
    cfg, device = model.config, model.device
    rng = np.random.default_rng(0)

    card = model.tokenizer.encode(f"<|im_start|>system\n{PROMPT}<|im_end|>\n", add_special_tokens=False).ids
    n_card, n = len(card), len(card) + PREFILL_AUDIO + STEPS
    text = rng.integers(1000, 5000, size=n)
    text[:n_card] = card
    text[n_card : n_card + PREFILL_AUDIO] = cfg.pad_token_id
    model_latents = rng.standard_normal((n, cfg.latent_dim)).astype(np.float32) * 2.0
    user_latents = rng.standard_normal((n, cfg.latent_dim)).astype(np.float32) * 2.0
    positions = np.arange(n)
    model_mask = positions >= n_card
    user_mask = positions >= n_card + PREFILL_AUDIO
    noise = rng.standard_normal((n, cfg.latent_dim)).astype(np.float32)
    clip = (0.1 * rng.standard_normal(CLIP_SAMPLES)).astype(np.float32)
    frames = (0.1 * rng.standard_normal((1, 3 * cfg.frame_samples))).astype(np.float32)
    latents = rng.standard_normal((1, 3, cfg.latent_dim)).astype(np.float32) * 2.0

    def tensor(array):
        return torch.as_tensor(array, device=device)

    # Backbone: the prompt in one call, then one position at a time.
    cache = model.new_cache()
    prefill = n - STEPS
    spans = [(0, prefill)] + [(i, i + 1) for i in range(prefill, n)]
    hidden = []
    for s, e in spans:
        embeddings = model.embed(
            tensor(text[None, s:e]), tensor(model_latents[None, s:e]), tensor(user_latents[None, s:e]),
            tensor(model_mask[None, s:e]), tensor(user_mask[None, s:e]),
        )
        hidden.append(model(embeddings, cache)[0])
    hidden = torch.cat(hidden)

    stream_ids = tensor(np.array([cfg.pad_token_id, cfg.eop_token_id]))
    logits = model.logits(hidden[-STEPS:])
    head_latents = model.head.sample(hidden, tensor(noise).to(model.dtype), cfg.head.guidance_scale)

    stream = CodecStream(model.codec)
    np.savez(
        args.out,
        text=text, model_latents=model_latents, user_latents=user_latents,
        model_mask=model_mask, user_mask=user_mask, prefill=prefill, noise=noise,
        clip=clip, frames=frames, latents=latents,
        hidden=hidden.float().cpu().numpy(),
        logits_top=logits.topk(50, dim=-1).indices.cpu().numpy(),
        logits_max=logits.max(dim=-1).values.cpu().numpy(),
        stream_logits=model.logits_fp32(hidden[-STEPS:], stream_ids).cpu().numpy(),
        head_latents=model.unscale_latents(head_latents).cpu().numpy(),
        clip_mean=model.codec.encode(tensor(clip)).float().cpu().numpy(),
        clip_audio=model.codec.decode(tensor(latents)).float().cpu().numpy(),
        stream_mean=stream.encode(tensor(frames)).float().cpu().numpy(),
        stream_audio=torch.cat(
            [stream.decode(tensor(latents[:, i : i + 1])) for i in range(latents.shape[1])], dim=1
        ).float().cpu().numpy(),
    )
    print(f"wrote {args.out}: {n} positions ({prefill} prefill + {STEPS} steps)")


if __name__ == "__main__":
    main()
