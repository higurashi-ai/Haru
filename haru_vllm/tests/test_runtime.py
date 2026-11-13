"""Engine settings read from the environment (no vLLM needed)."""

import pytest
from haru_vllm.runtime import codec_options


def test_codec_options_default_to_auto():
    assert codec_options({}) == ("auto", None)
    assert codec_options({"HARU_CODEC_BACKEND": " ", "HARU_CODEC_NUM_STAGES": ""}) == ("auto", None)


def test_codec_options_from_environment():
    assert codec_options({"HARU_CODEC_BACKEND": "triton", "HARU_CODEC_NUM_STAGES": "2"}) == ("triton", 2)
    assert codec_options({"HARU_CODEC_BACKEND": "torch"}) == ("torch", None)


@pytest.mark.parametrize("environ", [{"HARU_CODEC_BACKEND": "cuda"}, {"HARU_CODEC_NUM_STAGES": "4"},
                                     {"HARU_CODEC_NUM_STAGES": "two"}])
def test_codec_options_reject_invalid_values(environ):
    with pytest.raises(ValueError):
        codec_options(environ)
