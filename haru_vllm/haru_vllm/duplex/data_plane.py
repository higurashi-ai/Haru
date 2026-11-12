"""Output projection: each frame's audio and text token into realtime deltas."""

from __future__ import annotations

from vllm_omni.engine.duplex.plugin import DuplexDataPlaneContext
from vllm_omni.model_executor.common.duplex.data_plane import CumulativeAudioTextDataPlane
from vllm_omni.model_executor.common.request_outputs import unwrap_request_output

from haru.sampling import StreamDetokenizer


class HaruDataPlane(CumulativeAudioTextDataPlane):
    """Audio arrives one frame per output; text is detokenized here from the frame's token."""

    audio_output_is_delta = True

    def __init__(self, encode_audio) -> None:
        super().__init__(encode_audio)
        self.tokenizer = None
        self.silent_ids: set[int] = set()
        self._detokenizers: dict[str, StreamDetokenizer] = {}

    def close_stream(self, request_id: str) -> None:
        super().close_stream(request_id)
        self._detokenizers.pop(request_id, None)

    def close_session(self, session_id: str, *, active_request_id: str | None = None) -> None:
        super().close_session(session_id, active_request_id=active_request_id)
        for request_id in [r for r in self._detokenizers if r not in self._requests]:
            del self._detokenizers[request_id]

    def _project_output(self, output: object, *, context: DuplexDataPlaneContext) -> dict[str, object] | None:
        projected = super()._project_output(output, context=context)
        request_output, completion = unwrap_request_output(output)
        request_id = getattr(request_output, "request_id", None)
        text = ""
        if self.tokenizer is not None and isinstance(request_id, str):
            detokenizer = self._detokenizers.get(request_id)
            if detokenizer is None:
                detokenizer = self._detokenizers[request_id] = StreamDetokenizer(self.tokenizer, self.silent_ids)
            for token in getattr(completion, "token_ids", None) or ():
                text += detokenizer.push(int(token))
        if projected is not None and text:
            projected["text"] = text
        return projected
