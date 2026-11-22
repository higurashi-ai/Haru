"""Shared by the tests that run a real checkpoint (``HARU_MODEL=/path/to/Haru-4B pytest -m gpu``)."""

import os

import pytest

from haru import codec_triton


def pytest_runtest_setup(item):
    """A ``gpu("triton")`` test also needs triton and a CUDA device."""
    mark = item.get_closest_marker("gpu")
    if mark is not None and "triton" in mark.args and not codec_triton.is_available():
        pytest.skip("needs triton and a CUDA device")


@pytest.fixture(scope="session")
def model():
    """The checkpoint named by ``HARU_MODEL``, loaded once for the whole run."""
    path = os.environ.get("HARU_MODEL")
    if not path:
        pytest.skip("set HARU_MODEL to a checkpoint")
    from haru import HaruModel

    return HaruModel.from_pretrained(path)
