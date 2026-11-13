"""Text-stream sampling: top-k / top-p, the DRY repetition penalty, detokenization."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DryConfig:
    """DRY penalty: discourage extending a token sequence that already occurred."""

    multiplier: float = 0.8
    base: float = 1.75
    allowed_length: int = 1
    lookback: int = 1024
    max_match_length: int = 32
    max_penalty: float = 32.0

    @property
    def history_size(self) -> int:
        return self.lookback + self.max_match_length + 1


def dry_penalties(history: list[int], config: DryConfig) -> dict[int, float]:
    """``{token: penalty}`` for tokens that would extend an earlier repeat."""
    n = len(history)
    if n < max(1, config.allowed_length):
        return {}
    longest: dict[int, int] = {}
    last = history[-1]
    for j in range(n - 2, max(0, n - 1 - config.lookback) - 1, -1):
        if history[j] != last:
            continue
        length = 1
        while (
            length < config.max_match_length
            and j - length >= 0
            and history[j - length] == history[n - 1 - length]
        ):
            length += 1
        if length >= config.allowed_length:
            candidate = history[j + 1]
            if length > longest.get(candidate, 0):
                longest[candidate] = length
    return {
        token: min(config.max_penalty, config.multiplier * config.base ** (length - config.allowed_length))
        for token, length in longest.items()
    }


class DryState:
    """Running DRY history of one text stream; ``exempt`` tokens never count."""

    def __init__(self, config: DryConfig, exempt: set[int]):
        self.config = config
        self.exempt = exempt
        self.history: list[int] = []
        self.penalties: dict[int, float] = {}

    def update(self, token: int) -> None:
        if token in self.exempt:
            return
        self.history.append(token)
        extra = len(self.history) - self.config.history_size
        if extra > 0:
            del self.history[:extra]
        self.penalties = {
            t: p for t, p in dry_penalties(self.history, self.config).items() if t not in self.exempt
        }

    def apply(self, logits: torch.Tensor) -> None:
        """Subtract the current penalties from ``logits [vocab]`` in place."""
        if self.penalties:
            ids = torch.tensor(list(self.penalties), device=logits.device)
            logits[ids] -= torch.tensor(list(self.penalties.values()), device=logits.device, dtype=logits.dtype)


def sample_token(
    logits: torch.Tensor,
    temperature: float = 0.7,
    top_k: int = 25,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """``logits [B, vocab]`` (float32) -> token ids ``[B]``. ``temperature <= 0`` is greedy."""
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if 0 < top_k < logits.shape[-1]:
        values, indices = torch.topk(logits, top_k, dim=-1)
    else:
        values, indices = torch.sort(logits, dim=-1, descending=True)
    probs = torch.softmax(values, dim=-1)
    if 0 < top_p < 1:
        keep = probs.cumsum(dim=-1) - probs < top_p
        probs = probs * keep
    choice = torch.multinomial(probs, 1, generator=generator)
    return indices.gather(-1, choice)[:, 0]


class StreamDetokenizer:
    """Incremental text of the model's text stream.

    A run of PAD / EOP (no text on those frames) is rendered as one space, and
    a multi-byte character is held back until all of its tokens have arrived.
    """

    def __init__(self, tokenizer, silent_ids: set[int]):
        self.tokenizer = tokenizer
        self.silent_ids = silent_ids
        self._pending: list[int] = []
        self._in_gap = False

    def push(self, token: int) -> str:
        if token in self.silent_ids:
            if self._in_gap:
                return ""
            self._in_gap = True
            return " "
        self._in_gap = False
        self._pending.append(token)
        text = self.tokenizer.decode(self._pending, skip_special_tokens=False)
        if text.endswith("�"):
            return ""
        self._pending.clear()
        return text
