"""Haru on vLLM-Omni: a full-duplex spoken dialogue model served over the realtime API.

Installing this package is enough: vLLM calls :func:`register` through the
``vllm.general_plugins`` entry point, after which

    vllm serve <checkpoint> --omni --deploy-config $(python -m haru_vllm.deploy)

serves ``/v1/realtime?duplex=1``.
"""

__version__ = "0.1.1"

ARCHITECTURE = "HaruForConditionalGeneration"
_registered = False


def register() -> None:
    """Register the Haru config, model architecture and pipeline with vLLM-Omni."""
    global _registered
    if _registered:
        return
    _registered = True

    from transformers import AutoConfig
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP, Qwen3_5ForCausalLMConfig
    from vllm_omni.config.pipeline_registry import register_pipeline
    from vllm_omni.model_executor.models.registry import OmniModelRegistry

    from .configuration import HaruVllmConfig
    from .pipeline import HARU_PIPELINE

    AutoConfig.register(HaruVllmConfig.model_type, HaruVllmConfig, exist_ok=True)
    target = "haru_vllm.model:HaruForConditionalGeneration"
    ModelRegistry.register_model(ARCHITECTURE, target)
    OmniModelRegistry.register_model(ARCHITECTURE, target)
    # The backbone is a text-only Qwen3.5: same hybrid-cache and position setup.
    MODELS_CONFIG_MAP.setdefault(ARCHITECTURE, Qwen3_5ForCausalLMConfig)
    register_pipeline(HARU_PIPELINE)
