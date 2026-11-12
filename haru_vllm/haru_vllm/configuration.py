"""Hugging Face config wrapper so vLLM can read a Haru ``config.json``."""

from __future__ import annotations

import inspect
from typing import Any

from transformers import PretrainedConfig, Qwen3_5TextConfig

from haru.config import HaruConfig


class HaruVllmConfig(PretrainedConfig):
    model_type = "haru"
    sub_configs = {"text_config": Qwen3_5TextConfig}

    def __init__(self, lm: dict[str, Any] | None = None, text_config=None, **kwargs):
        if isinstance(text_config, Qwen3_5TextConfig):
            self.text_config = text_config
        else:
            params = dict(text_config or lm or {})
            accepted = inspect.signature(Qwen3_5TextConfig.__init__).parameters
            self.text_config = Qwen3_5TextConfig(**{k: v for k, v in params.items() if k in accepted})
        self.lm = dict(lm or {})
        super().__init__(**kwargs)

    def get_text_config(self, *args, **kwargs) -> Qwen3_5TextConfig:
        return self.text_config

    @property
    def haru(self) -> HaruConfig:
        """The checkpoint configuration as the ``haru`` package reads it."""
        params = {k: v for k, v in self.to_dict().items() if k not in ("text_config", "model_type")}
        return HaruConfig.from_dict(params)
