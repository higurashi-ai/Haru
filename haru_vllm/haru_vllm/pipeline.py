"""Pipeline topology: one autoregressive stage that also produces the audio.

The audio latent of a frame is an input of the next frame, so the MeanFlow head
and the codec run inside the language-model stage, right after sampling.
"""

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

HARU_PIPELINE = PipelineConfig(
    model_type="haru",
    model_arch="HaruForConditionalGeneration",
    duplex_plugin="haru_vllm.duplex.plugin.HaruDuplexPlugin",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="haru",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            owns_tokenizer=True,
            sampling_constraints={"detokenize": False},
            # Conversation state lives in the worker and cannot be recomputed.
            recompute_preemption="fail",
        ),
    ),
)
