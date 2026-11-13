import numpy as np
import pytest

from haru.playout import PACKET_SAMPLES, PlayoutBuffer, PlayoutOptions, upsample_2x, wsola_compress, UplinkBuffer, UplinkOptions

FRAME = 3200  # one model frame at 24 kHz


def speech_like(samples: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(samples) / 24_000
    return (0.3 * np.sin(2 * np.pi * 180 * t) + 0.02 * rng.standard_normal(samples)).astype(np.float32)


def test_upsample_doubles_the_length_and_interpolates():
    up = upsample_2x(np.array([0.0, 0.5, 1.0], dtype=np.float32))
    assert up.dtype == np.int16
    assert up.tolist() == [0, 8191, 16383, 24575, 32767, 32767]


def test_wsola_shortens_and_keeps_both_ends():
    x = upsample_2x(speech_like(FRAME))
    y = wsola_compress(x, 0.2)
    assert len(y) == round(len(x) * 0.8)
    assert np.array_equal(y[:100], x[:100])
    assert y[-1] == x[-1]
    # A steady tone keeps its level: windows are aligned, not just overlapped.
    assert np.abs(y).mean() == pytest.approx(np.abs(x).mean(), rel=0.1)


def test_wsola_leaves_short_or_unstretched_audio_alone():
    x = upsample_2x(speech_like(FRAME))
    assert wsola_compress(x, 0.0) is x
    assert wsola_compress(x[:1000], 0.2) is not None and len(wsola_compress(x[:1000], 0.2)) == 1000


def test_buffer_sends_silence_until_started_then_audio():
    buffer = PlayoutBuffer()
    buffer.push(speech_like(FRAME))
    assert not buffer.pop().any() and buffer.underrun_ms == 0
    buffer.start()
    packets = [buffer.pop() for _ in range(8)]
    assert all(len(p) == PACKET_SAMPLES for p in packets)
    assert all(p.any() for p in packets[:6])  # 133 ms = 6 whole packets
    assert not packets[7].any()
    assert buffer.sent_ms == 120.0
    assert buffer.underrun_ms == 40.0


def test_buffer_catches_up_after_a_burst_without_dropping():
    buffer = PlayoutBuffer(PlayoutOptions(start_margin_ms=40.0, max_buffer_ms=1000.0))
    buffer.push(speech_like(FRAME))
    buffer.start()
    buffer.push(speech_like(FRAME, 1))  # arrives with a whole frame still queued
    assert buffer.compressed_ms == pytest.approx(133.3 * 0.10, abs=1.0)
    assert buffer.dropped_ms == 0


def test_buffer_trims_a_queue_beyond_the_cap():
    buffer = PlayoutBuffer(PlayoutOptions(max_buffer_ms=320.0, keep_ms=140.0, catchup_rate=0.0, catchup_fast_rate=0.0))
    buffer.start()
    for seed in range(3):
        buffer.push(speech_like(FRAME, seed))
    assert buffer.queued_ms == pytest.approx(140.0)
    assert buffer.dropped_ms == pytest.approx(260.0)


def _packets(count, value=1.0):
    return [np.full(480, value, dtype=np.float32) for _ in range(count)]


def test_uplink_passes_audio_through_in_order():
    buffer = UplinkBuffer(UplinkOptions(playout_ms=40))
    assert not buffer.ready
    buffer.add(0, _packets(2, 1.0), now=0.0)
    assert buffer.ready
    buffer.add(2, _packets(1, 2.0), now=0.02)
    assert [float(buffer.take(0.02 * i)[0]) for i in range(3)] == [1.0, 1.0, 2.0]
    assert buffer.concealed_ms == 0 and buffer.lost_ms == 0


def test_uplink_fills_lost_packets_with_silence():
    buffer = UplinkBuffer(UplinkOptions(playout_ms=20))
    buffer.add(0, _packets(1), now=0.0)
    buffer.add(3, _packets(1), now=0.06)  # slots 1 and 2 never arrived
    out = [float(buffer.take(0.02 * i)[0]) for i in range(4)]
    assert out == [1.0, 0.0, 0.0, 1.0]
    assert buffer.lost_ms == 40 and buffer.concealed_ms == 40


def test_uplink_grows_for_late_packets_then_shrinks():
    buffer = UplinkBuffer(UplinkOptions(playout_ms=20, playout_max_ms=40, shrink_after_s=1.0))
    buffer.add(0, _packets(1), now=0.0)
    assert float(buffer.take(0.0)[0]) == 1.0
    assert float(buffer.take(0.02)[0]) == 0.0  # slot 1 is late: silence goes out
    buffer.add(1, _packets(1, 2.0), now=0.03)  # it arrives after all: kept, delay grows
    assert buffer.grown == 1 and buffer.target_ms == 40
    assert float(buffer.take(0.04)[0]) == 2.0
    assert float(buffer.take(0.06)[0]) == 0.0
    buffer.add(2, _packets(1, 3.0), now=0.07)  # late again, already at the limit: discarded
    assert buffer.late_ms == 20 and buffer.target_ms == 40
    buffer.add(3, _packets(2, 4.0), now=0.08)
    assert float(buffer.take(2.0)[0]) == 4.0  # quiet for over a second: one slot is given back
    assert buffer.target_ms == 20 and buffer.trimmed_ms == 20


def test_uplink_trims_a_backlog():
    buffer = UplinkBuffer(UplinkOptions(playout_ms=20, max_extra_ms=40))
    buffer.add(0, _packets(10), now=0.0)  # a stall delivered 200 ms at once
    buffer.take(0.0)
    assert buffer.trimmed_ms == 180
