// @ts-nocheck
function asMs(samples) {
  return (samples * 1000 / sampleRate).toFixed(1);
}

function asSamples(mili) {
  return Math.round(mili * sampleRate / 1000);
}

// Shorten ``x`` by ``rate`` (0..0.25) without changing its pitch (WSOLA): Hann windows of ``win`` samples at 50 % overlap, each taken near its nominal
// (faster) input position where it best continues the previous one (cross-correlation
// within +-``tol``). Starts with the input's first samples and crossfades into its last
// ``win / 2`` so consecutive chunks still join seamlessly.
function wsolaCompress(x, rate, win, tol) {
  const n = x.length;
  const outLen = Math.round(n * (1 - rate));
  const hop = win >> 1;
  if (rate <= 0 || n < 4 * win || outLen <= win) {
    return x;
  }
  const speed = n / outLen;
  const out = new Float32Array(outLen + win);
  const wsum = new Float32Array(outLen + win);
  const w = new Float32Array(win);
  for (let i = 0; i < win; i++) {
    w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / win);
  }
  // First window: flat over its first half, so the output starts with the input itself.
  for (let i = 0; i < win; i++) {
    const wi = i < hop ? 1 : w[i];
    out[i] += x[i] * wi;
    wsum[i] += wi;
  }
  let prev = 0;
  for (let k = 1; k * hop + hop <= outLen; k++) {
    const nominal = Math.round(k * hop * speed);
    const natural = prev + hop;  // what would follow the previous window unchanged
    const hi = Math.min(n - win, nominal + tol);
    const lo = Math.min(Math.max(0, nominal - tol), hi);  // near the end: the last window that fits
    let best = nominal >= lo && nominal <= hi ? nominal : lo;
    if (natural + win <= n) {
      let bestCorr = -Infinity;
      for (let c = lo; c <= hi; c++) {
        let corr = 0;
        for (let i = 0; i < win; i++) {
          corr += x[c + i] * x[natural + i];
        }
        if (corr > bestCorr) {
          bestCorr = corr;
          best = c;
        }
      }
    }
    for (let i = 0; i < win; i++) {
      out[k * hop + i] += x[best + i] * w[i];
      wsum[k * hop + i] += w[i];
    }
    prev = best;
  }
  const y = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    y[i] = wsum[i] > 1e-3 ? out[i] / wsum[i] : 0;
  }
  // Last hop: crossfade into the input's own tail (ending exactly on its last sample).
  for (let i = 0; i < hop; i++) {
    const r = (i + 1) / hop;
    y[outLen - hop + i] = y[outLen - hop + i] * (1 - r) + x[n - hop + i] * r;
  }
  return y;
}

class PlayoutProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Buffer length definitions
    // The model emits 133.3 ms frames (3200 samples at 24 kHz), so every threshold must
    // fit at least one whole frame or the drop logic fires on every packet.
    let frameSize = asSamples(400 / 3);
    // initialBufferSamples: we wait to have at least that many samples before starting to play
    this.initialBufferSamples = 2 * frameSize;
    // once we have enough samples, we further wait that long before starting to play.
    // This allows to have buffer lengths that are not a multiple of frameSize.
    this.partialBufferSamples = asSamples(20);
    // If the buffer length goes over that many, we will drop the oldest packets until
    // we reach back initialBufferSamples + partialBufferSamples.
    this.maxBufferSamples = frameSize;
    // increments
    this.partialBufferIncrement = asSamples(20);
    this.maxPartialWithIncrements = asSamples(200);
    this.maxBufferSamplesIncrement = frameSize;
    this.maxMaxBufferWithIncrements = asSamples(400);
    // Catch-up: after a stall the restart margin and the burst that
    // follows leave more audio queued than an undisturbed stream keeps (one frame plus 100 ms,
    // measured at its lowest, just before a frame arrives; +20 ms missed ~3 % of the audio
    // on a jittery uplink, as the steady queue left no margin for the late frames). While the
    // lowest queue of the last ~0.7 s is more than CATCHUP_DEADBAND above that, each
    // arriving chunk is shortened by WSOLA: up to 10 %, 20 % while the excess is above
    // 150 ms. The removed time counts as played stream time, like a drop.
    this.catchupTarget = frameSize + asSamples(100);
    this.catchupDeadband = asSamples(20);
    this.catchupFastAbove = asSamples(150);
    this.catchupWin = asSamples(10);
    this.catchupTol = asSamples(2.5);
    this.minWindowSamples = asSamples(333);
    this.statsEverySamples = asSamples(20);
    this.statsLeft = this.statsEverySamples;

    // State and metrics
    this.initState();

    this.port.onmessage = (event) => {
      if (event.data.type == "reset") {
        // Per connection, from the server's network profile (playoutProfile.ts).
        if (event.data.startMs) {
          this.initialBufferSamples = asSamples(event.data.startMs);
        }
        if (event.data.catchupTargetMs) {
          this.catchupTarget = asSamples(event.data.catchupTargetMs);
        }
        this.initState();
        return;
      }
      let frame = event.data.frame;
      if (this.started && this.actualAudioPlayed > 0) {
        const excess = this.recentMinQueue() - this.catchupTarget;
        if (excess > this.catchupDeadband) {
          const cap = excess > this.catchupFastAbove ? 0.20 : 0.10;
          const short = wsolaCompress(frame, Math.min(cap, excess / frame.length), this.catchupWin, this.catchupTol);
          const removed = frame.length - short.length;
          if (removed > 0) {
            this.compressedAudio += removed / sampleRate;
            this.timeInStream += removed / sampleRate;
            frame = short;
          }
        }
      }
      this.frames.push(frame);
      if (this.currentSamples() >= this.initialBufferSamples && !this.started) {
        this.start();
      }
      if (this.currentSamples() >= this.totalMaxBufferSamples()) {
        let target = this.initialBufferSamples + this.partialBufferSamples
        while (this.currentSamples() > (this.initialBufferSamples + this.partialBufferSamples)) {
          let first = this.frames[0];
          let to_remove = this.currentSamples() - target;
          to_remove = Math.min(first.length - this.offsetInFirstBuffer, to_remove);
          this.offsetInFirstBuffer += to_remove;
          this.timeInStream += to_remove / sampleRate;
          if (this.offsetInFirstBuffer == first.length) {
            this.frames.shift();
            this.offsetInFirstBuffer = 0;
          }
        }
        this.maxBufferSamples += this.maxBufferSamplesIncrement;
        this.maxBufferSamples = Math.min(this.maxMaxBufferWithIncrements, this.maxBufferSamples);
      }
      this.postStats();
    };
  }

  // Also posted every ~20 ms from process(): the main thread takes the latency as mic time
  // minus the played stream time, both extrapolated to the moment it reads them (a value
  // computed only when a model frame arrives is phase-locked to the frames).
  postStats() {
    this.statsLeft = this.statsEverySamples;
    this.port.postMessage({
      totalAudioPlayed: this.totalAudioPlayed,
      actualAudioPlayed: this.actualAudioPlayed,
      timeInStream: this.timeInStream,
      playing: this.canPlay(),
      at: currentTime,
      minDelay: this.minDelay,
      maxDelay: this.maxDelay,
      compressedAudio: this.compressedAudio,
    });
  }

  initState() {
    this.frames = new Array();
    this.offsetInFirstBuffer = 0;
    this.firstOut = false;
    this.remainingPartialBufferSamples = 0;
    this.timeInStream = 0.;
    this.resetStart();

    // Metrics
    this.totalAudioPlayed = 0.;
    this.actualAudioPlayed = 0.;
    this.maxDelay = 0.;
    this.minDelay = 2000.;
    this.compressedAudio = 0.;
    // Lowest queue in the current and the previous ~333 ms window (catch-up).
    this.minQueueCur = Infinity;
    this.minQueuePrev = Infinity;
    this.minWindowLeft = 0;
    // For now let's reset the buffer params.
    this.partialBufferSamples = asSamples(20);
    this.maxBufferSamples = asSamples(400 / 3);
  }

  totalMaxBufferSamples() {
    return this.maxBufferSamples + this.partialBufferSamples + this.initialBufferSamples;
  }

  timestamp() {
    return Date.now() % 1000;
  }

  currentSamples() {
    let samples = 0;
    for (let k = 0; k < this.frames.length; k++) {
      samples += this.frames[k].length
    }
    samples -= this.offsetInFirstBuffer;
    return samples;
  }

  recentMinQueue() {
    return Math.min(this.minQueueCur, this.minQueuePrev);
  }

  trackMinQueue(queued, played) {
    this.minQueueCur = Math.min(this.minQueueCur, queued);
    this.minWindowLeft -= played;
    if (this.minWindowLeft <= 0) {
      this.minQueuePrev = this.minQueueCur;
      this.minQueueCur = Infinity;
      this.minWindowLeft = this.minWindowSamples;
    }
  }

  resetStart() {
    this.started = false;
  }

  start() {
    this.started = true;
    this.remainingPartialBufferSamples = this.partialBufferSamples;
    this.firstOut = true;
  }

  canPlay() {
    return this.started && this.frames.length > 0 && this.remainingPartialBufferSamples <= 0;
  }

  process(inputs, outputs, parameters) {
    let delay = this.currentSamples() / sampleRate;
    if (this.canPlay()) {
      this.maxDelay = Math.max(this.maxDelay, delay);
      this.minDelay = Math.min(this.minDelay, delay);
    }
    const output = outputs[0][0];
    if (!this.canPlay()) {
      if (this.actualAudioPlayed > 0) {
        this.totalAudioPlayed += output.length / sampleRate;
      }
      this.remainingPartialBufferSamples -= output.length;
      this.statsLeft -= output.length;
      if (this.statsLeft <= 0) {
        this.postStats();
      }
      return true;
    }
    this.trackMinQueue(this.currentSamples(), output.length);
    let first = this.frames[0];
    let out_idx = 0;
    while (out_idx < output.length && this.frames.length) {
      let first = this.frames[0];
      let to_copy = Math.min(first.length - this.offsetInFirstBuffer, output.length - out_idx);
      output.set(first.subarray(this.offsetInFirstBuffer, this.offsetInFirstBuffer + to_copy), out_idx);
      this.offsetInFirstBuffer += to_copy;
      out_idx += to_copy;
      if (this.offsetInFirstBuffer == first.length) {
        this.offsetInFirstBuffer = 0;
        this.frames.shift();
      }
    }
    if (this.firstOut) {
      this.firstOut = false;
      for (let i = 0; i < out_idx; i++) {
        output[i] *= i / out_idx;
      }
    }
    if (out_idx < output.length) {
      this.partialBufferSamples += this.partialBufferIncrement;
      this.partialBufferSamples = Math.min(this.partialBufferSamples, this.maxPartialWithIncrements);
      // We ran out of a buffer, let's revert to the started state to replenish it.
      this.resetStart();
      for (let i = 0; i < out_idx; i++) {
        output[i] *= (out_idx - i) / out_idx;
      }
    }
    this.totalAudioPlayed += output.length / sampleRate;
    this.actualAudioPlayed += out_idx / sampleRate;
    this.timeInStream += out_idx / sampleRate;
    this.statsLeft -= output.length;
    if (this.statsLeft <= 0) {
      this.postStats();
    }
    return true;
  }
}
registerProcessor("haru-processor", PlayoutProcessor);
