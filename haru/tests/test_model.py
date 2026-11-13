"""End-to-end checks on a real checkpoint: ``HARU_MODEL=/path/to/Haru-4B pytest -m gpu``."""

import asyncio
import os

import numpy as np
import pytest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not os.environ.get("HARU_MODEL"), reason="set HARU_MODEL to a checkpoint"),
]

FRAME = 3200


@pytest.fixture(scope="module")
def model():
    from haru import HaruModel

    return HaruModel.from_pretrained(os.environ["HARU_MODEL"])


def voice(seconds: float = 4.0) -> np.ndarray:
    t = np.arange(int(seconds * 24_000)) / 24_000
    return (0.2 * np.sin(2 * np.pi * 150 * t) * (1 + np.sin(2 * np.pi * 3 * t))).astype(np.float32)


def test_session_produces_frames_and_is_reproducible(model):
    from haru import DuplexConfig, DuplexSession

    def run() -> list:
        config = DuplexConfig(seed=7, speak_first=True, max_frames=12)
        session = DuplexSession(model, "You are Haru. Greet the other person.", voice(), config)
        frames = [session.start()]
        while not frames[-1].done:
            frames.append(session.step(np.zeros(FRAME, dtype=np.float32)))
        return frames

    first, second = run(), run()
    assert len(first) == 12 and [f.index for f in first] == list(range(12))
    assert all(f.audio.shape == (FRAME,) and f.audio.dtype == np.float32 for f in first)
    assert all(np.isfinite(f.audio).all() and np.abs(f.audio).max() <= 1.0 for f in first)
    assert [f.token for f in first] == [f.token for f in second]
    assert first[1].token not in (model.config.pad_token_id, model.config.eop_token_id)  # speak_first
    assert "".join(f.text for f in first).strip()


def test_step_rejects_a_wrong_frame_size(model):
    from haru import DuplexSession

    session = DuplexSession(model)
    with pytest.raises(RuntimeError):
        session.step(np.zeros(FRAME, dtype=np.float32))
    session.start()
    with pytest.raises(ValueError):
        session.step(np.zeros(FRAME - 1, dtype=np.float32))


def test_local_backend_runs_off_the_event_loop(model):
    from haru.backends import LocalBackend, SessionSettings

    async def scenario():
        backend = LocalBackend(os.environ["HARU_MODEL"], warmup_frames=0)
        backend.model = model  # already loaded by the fixture
        conversation = await backend.open(SessionSettings("You are Haru.", voice(), max_frames=4, seed=0))
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.001)
                ticks += 1

        task = asyncio.create_task(ticker())
        frames = [await conversation.start()]
        while not frames[-1].done:
            frames.append(await conversation.step(np.zeros(FRAME, dtype=np.float32)))
        task.cancel()
        await conversation.close()
        return frames, ticks

    frames, ticks = asyncio.run(scenario())
    assert len(frames) == 4
    assert ticks > 0  # the loop kept running while frames were computed
