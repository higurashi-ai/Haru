"""Paced playout of model audio for the WebRTC path.

The model produces 133 ms of audio at a time; a peer connection sends 20 ms
packets on a fixed clock. :class:`PlayoutBuffer` sits between the two: it
upsamples to 48 kHz, hands out one packet per tick, sends silence while the
model is late, and brings the latency back after a burst by time-stretching
(WSOLA) instead of dropping audio.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

RTC_SAMPLE_RATE = 48_000
PACKET_SAMPLES = 960  # 20 ms at 48 kHz, one RTP packet


def wsola_compress(x: np.ndarray, rate: float, window: int = PACKET_SAMPLES, tolerance: int = 240) -> np.ndarray:
    """Shorten int16 audio by the fraction ``rate`` (0..0.25) without changing its pitch.

    Hann windows of ``window`` samples at 50 % overlap; each is taken near its
    nominal (faster) input position, where it best continues the previous one
    (cross-correlation within ``tolerance`` samples). The output starts with the
    input's own samples and crossfades into its tail, so consecutive chunks
    still join seamlessly.
    """
    n = int(x.shape[0])
    out_len = int(round(n * (1.0 - rate)))
    hop = window // 2
    if rate <= 0 or n < 4 * window or out_len <= window:
        return x
    xf = x.astype(np.float32)
    speed = n / out_len
    hann = np.hanning(window + 1)[:window].astype(np.float32)
    out = np.zeros(out_len + window, dtype=np.float32)
    weight = np.zeros(out_len + window, dtype=np.float32)
    first = hann.copy()
    first[:hop] = 1.0
    out[:window] += xf[:window] * first
    weight[:window] += first
    previous, k = 0, 1
    while k * hop + hop <= out_len:
        nominal = int(round(k * hop * speed))
        natural = previous + hop  # what would follow the previous window unchanged
        high = min(n - window, nominal + tolerance)
        low = min(max(0, nominal - tolerance), high)
        best = nominal if low <= nominal <= high else low
        if natural + window <= n:
            correlation = np.correlate(xf[low : high + window], xf[natural : natural + window], mode="valid")
            best = low + int(np.argmax(correlation))
        out[k * hop : k * hop + window] += xf[best : best + window] * hann
        weight[k * hop : k * hop + window] += hann
        previous = best
        k += 1
    y = np.where(weight[:out_len] > 1e-3, out[:out_len] / np.maximum(weight[:out_len], 1e-3), 0.0)
    ramp = np.arange(1, hop + 1, dtype=np.float32) / hop
    y[-hop:] = y[-hop:] * (1.0 - ramp) + xf[-hop:] * ramp
    return np.clip(y, -32768, 32767).astype(np.int16)


def upsample_2x(pcm: np.ndarray) -> np.ndarray:
    """float32 24 kHz in [-1, 1] -> int16 48 kHz (linear interpolation, enough for speech into Opus)."""
    x = np.asarray(pcm, dtype=np.float32) * 32767.0
    up = np.empty(2 * x.shape[0], dtype=np.float32)
    up[0::2] = x
    up[1::2] = np.concatenate([(x[:-1] + x[1:]) * 0.5, x[-1:]])
    return np.clip(up, -32768, 32767).astype(np.int16)


@dataclass
class PlayoutOptions:
    start_margin_ms: float = 40.0  # wait after the first audio; also the queue level catch-up returns to
    max_buffer_ms: float = 320.0  # above this the oldest audio is dropped ...
    keep_ms: float = 140.0  # ... back to this much
    catchup_deadband_ms: float = 10.0
    catchup_rate: float = 0.10  # at most this fraction of a frame is removed ...
    catchup_fast_rate: float = 0.20  # ... or this, while the excess is above
    catchup_fast_above_ms: float = 150.0


class PlayoutBuffer:
    """Queue of 48 kHz int16 model audio, read one 20 ms packet at a time.

    ``push`` adds a model frame; ``pop`` returns the next packet, silence while
    nothing is queued (an underrun: playback resumes as soon as audio is back).
    When frames arrive in a burst the queue grows; each following frame is then
    shortened with WSOLA until the queue is back at ``start_margin_ms``. A
    queue beyond ``max_buffer_ms`` is trimmed outright.
    """

    def __init__(self, options: PlayoutOptions | None = None):
        self.options = options or PlayoutOptions()
        self._queue = np.zeros(0, dtype=np.int16)
        self.started = False
        self.sent_ms = 0.0
        self.underrun_ms = 0.0
        self.dropped_ms = 0.0
        self.compressed_ms = 0.0

    @property
    def queued_ms(self) -> float:
        return 1000.0 * self._queue.shape[0] / RTC_SAMPLE_RATE

    def start(self) -> None:
        """Begin playback, keeping at most ``keep_ms`` of what was queued while connecting."""
        self.started = True
        self._trim(self.options.keep_ms)

    def push(self, pcm24: np.ndarray) -> None:
        """Queue one model frame (float32, 24 kHz)."""
        opts = self.options
        audio = upsample_2x(pcm24)
        frame_ms = 1000.0 * audio.shape[0] / RTC_SAMPLE_RATE
        excess_ms = self.queued_ms - opts.start_margin_ms
        if self.started and excess_ms > opts.catchup_deadband_ms and frame_ms > 0:
            cap = opts.catchup_fast_rate if excess_ms > opts.catchup_fast_above_ms else opts.catchup_rate
            shorter = wsola_compress(audio, min(cap, excess_ms / frame_ms))
            self.compressed_ms += 1000.0 * (audio.shape[0] - shorter.shape[0]) / RTC_SAMPLE_RATE
            audio = shorter
        self._queue = np.concatenate([self._queue, audio])
        if self.queued_ms > opts.max_buffer_ms:
            self._trim(opts.keep_ms)

    def pop(self) -> np.ndarray:
        """The next 20 ms packet (int16, 48 kHz)."""
        if self.started and self._queue.shape[0] >= PACKET_SAMPLES:
            packet, self._queue = self._queue[:PACKET_SAMPLES], self._queue[PACKET_SAMPLES:]
            self.sent_ms += 20.0
            return packet
        if self.started:
            self.underrun_ms += 20.0
        return np.zeros(PACKET_SAMPLES, dtype=np.int16)

    def _trim(self, keep_ms: float) -> None:
        keep = int(keep_ms * RTC_SAMPLE_RATE / 1000)
        extra = self._queue.shape[0] - keep
        if extra > 0:
            self.dropped_ms += 1000.0 * extra / RTC_SAMPLE_RATE
            self._queue = self._queue[extra:]


@dataclass
class UplinkOptions:
    playout_ms: float = 40.0  # microphone audio queued before feeding starts
    playout_max_ms: float = 200.0  # the queue target may grow to this when packets arrive late
    shrink_after_s: float = 15.0  # ... and gives 20 ms back after this long without a late packet
    max_extra_ms: float = 200.0  # a queue this far above its target is trimmed
    max_gap_fill_ms: float = 1000.0  # longest run of lost packets replaced by silence


class UplinkBuffer:
    """Adaptive jitter buffer for the user's microphone: 20 ms packets in, one slot per 20 ms out.

    The model consumes user audio on a fixed clock, so the timeline must stay
    continuous. ``add`` queues arriving packets and marks lost ones as silence;
    ``take`` returns the slot that is due. A slot with nothing queued goes out
    as silence; when its packet turns up afterwards the target delay grows by
    20 ms (up to ``playout_max_ms``) and the packet is kept, otherwise it is
    discarded. After ``shrink_after_s`` without lateness the target shrinks again.
    """

    def __init__(self, options: UplinkOptions | None = None, slot_samples: int = 480):
        self.options = opts = options or UplinkOptions()
        self._silence = np.zeros(slot_samples, dtype=np.float32)
        self._fifo: deque[np.ndarray | None] = deque()  # None: a lost packet
        self._min_target = self._target = max(1, int(opts.playout_ms // 20))
        self._max_target = max(self._target, int(opts.playout_max_ms // 20))
        self._extra = max(1, int(opts.max_extra_ms // 20))
        self._max_gap = int(opts.max_gap_fill_ms // 20)
        self._debt = 0  # slots sent as silence whose packets may still arrive
        self._next_slot: int | None = None
        self._last_late = 0.0
        self.concealed_ms = 0.0
        self.lost_ms = 0.0
        self.late_ms = 0.0
        self.trimmed_ms = 0.0
        self.grown = 0

    @property
    def ready(self) -> bool:
        """Enough audio is queued to start feeding."""
        return len(self._fifo) >= self._target

    @property
    def target_ms(self) -> float:
        return 20.0 * self._target

    def add(self, slot: int, packets: list[np.ndarray], now: float) -> None:
        """Queue ``packets`` (20 ms each); ``slot`` is the first one's index on the sender's 20 ms clock."""
        if self._next_slot is not None:
            gap = slot - self._next_slot
            if 0 < gap <= self._max_gap:
                # Lost packets: slots the feeder has not already covered with silence.
                self.lost_ms += 20.0 * gap
                covered = min(gap, self._debt)
                self._debt -= covered
                self._fifo.extend([None] * (gap - covered))
        self._next_slot = slot + len(packets)
        for packet in packets:
            if self._debt == 0:
                self._fifo.append(packet)
                continue
            self._debt -= 1  # its slot already went out as silence
            self._last_late = now
            if self._target < self._max_target:
                self._target += 1
                self._fifo.append(packet)
                self.grown += 1
            else:
                self.late_ms += 20.0

    def take(self, now: float) -> np.ndarray:
        """The 20 ms slot due now."""
        if self._target > self._min_target and now - self._last_late > self.options.shrink_after_s:
            self._target -= 1
            self._last_late = now
            if len(self._fifo) > self._target:
                self._fifo.popleft()
                self.trimmed_ms += 20.0
        if len(self._fifo) > self._target + self._extra:
            drop = len(self._fifo) - self._target
            for _ in range(drop):
                self._fifo.popleft()
            self.trimmed_ms += 20.0 * drop
        if self._fifo:
            packet = self._fifo.popleft()
            if packet is not None:
                return packet
        else:
            self._debt += 1
        self.concealed_ms += 20.0
        return self._silence
