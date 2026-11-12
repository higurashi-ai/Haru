// Haru's orbs and waveform lanes: two lanes on one clock, newest at the right, each with a living orb.

export const FONT = '-apple-system, BlinkMacSystemFont, "SF Pro Text", "Hiragino Sans", "Hiragino Kaku Gothic ProN", "Noto Sans JP", "Yu Gothic UI", "Segoe UI", sans-serif';

export const Theme = {
  ink: "#1d1d1f",
  muted: "#6e6e73",
  line: "#d2d2d7",
  paper: "#ffffff",
  accent: "#e04a2f",
};

// Warm: you in a yellowish jade green, the model in coral to amber.
export const Palette = {
  user: ["#a3d977", "#3fb984", "#e8d36a"],
  model: ["#ff9a7b", "#f9614f", "#ffb35c"],
  userBars: "#3aa877",
  modelBars: "#f26b4f",
  lane: "#f4f5f7",
};

// The exported video: drawn in a 720-unit layout at 1.5x (1080x1080).
export const VIDEO_LAYOUT = 720;
export const VIDEO_SCALE = 1.5;
export const VIDEO_PIXELS = VIDEO_LAYOUT * VIDEO_SCALE;

export const now = () => performance.now() / 1000;

// ---- History

export type Track = 0 | 1;
export const USER: Track = 0;
export const MODEL: Track = 1;

// Loudness per lane, 20 bins a second, on the page clock (`now()`).
export class WaveHistory {
  static binSeconds = 0.05;
  static capacity = 1200;  // one minute

  private peaks = [new Float32Array(WaveHistory.capacity), new Float32Array(WaveHistory.capacity)];
  private last: (number | null)[] = [null, null];  // newest bin written per lane

  private static slot(bin: number) {
    const c = WaveHistory.capacity;
    return ((bin % c) + c) % c;
  }

  add(track: Track, time: number, peak: number) {
    const bin = Math.floor(time / WaveHistory.binSeconds);
    const p = this.peaks[track];
    const newest = this.last[track];
    if (newest == null || bin - newest >= WaveHistory.capacity) {
      p.fill(0);
      this.last[track] = bin;
    } else if (bin > newest) {
      for (let b = newest + 1; b <= bin; b++) {
        p[WaveHistory.slot(b)] = 0;
      }
      this.last[track] = bin;
    } else if (newest - bin >= WaveHistory.capacity) {
      return;
    }
    const s = WaveHistory.slot(bin);
    p[s] = Math.max(p[s], peak);
  }

  // Peak of one bin (0 where nothing was heard).
  peak(track: Track, bin: number) {
    const newest = this.last[track];
    if (newest == null || bin > newest || newest - bin >= WaveHistory.capacity) {
      return 0;
    }
    return this.peaks[track][WaveHistory.slot(bin)];
  }
}

// 0...1 on a -60..0 dBFS scale.
export const normalizedLevel = (peak: number) =>
  Math.max(0, Math.min(1, (20 * Math.log10(Math.max(peak, 1e-6)) + 60) / 60));

// ---- Orb

// Smoothing factor for a time constant, independent of the frame rate.
const smoothing = (dt: number, tau: number) => 1 - Math.exp(-dt / tau);

export class BlobState {
  level = 0;
  phase = Math.random() * 10;
  lastTime: number | null = null;
}

const rgba = (hex: string, alpha: number) => {
  const v = parseInt(hex.slice(1), 16);
  return `rgba(${(v >> 16) & 255}, ${(v >> 8) & 255}, ${v & 255}, ${alpha})`;
};

// A soft, always-moving blob in the box (x, y, w, h): at rest it breathes, ripples
// and turns on its own; with the voice it swells, wobbles harder and turns faster.
export const drawBlob = (
  ctx: CanvasRenderingContext2D, x: number, y: number, width: number, height: number,
  raw: number, t: number, state: BlobState, colors: string[],
) => {
  const dt = Math.min(0.1, state.lastTime == null ? 0 : t - state.lastTime);
  state.lastTime = t;
  // Background noise leaves it at rest.
  const target = Math.max(0, Math.min(1, (raw - 0.2) / 0.6));
  state.level += (target - state.level) * smoothing(dt, target > state.level ? 0.05 : 0.25);
  const l = state.level;
  state.phase += dt * (1.7 + 2.2 * l);

  const w = Math.min(width, height);
  const cx = x + width / 2, cy = y + height / 2;
  const breath = 1 + 0.06 * Math.sin(t * 1.6) * (1 - l);
  const r = w * 0.22 * (1 + 0.4 * l) * breath;
  const amp = 0.025 + 0.045 * l;

  const blob = (radius: number, a: number, shift: number) => {
    const n = 96, ph = state.phase + shift;
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      const th = (i / n) * 2 * Math.PI;
      const k = 1 + a * (0.6 * Math.sin(2 * th + 0.7 * ph + 4.1) + 0.45 * Math.sin(3 * th + 1.3 * ph)
        + 0.15 * Math.sin(5 * th - 1.1 * ph + 1.7));
      const px = cx + radius * k * Math.cos(th), py = cy + radius * k * Math.sin(th);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.closePath();
  };

  blob(r * 1.2, amp * 1.2, 2.1);
  ctx.fillStyle = rgba(colors[2], 0.14 + 0.16 * l);
  ctx.fill();
  blob(r * 1.08, amp * 1.1, 1.0);
  ctx.fillStyle = rgba(colors[0], 0.26 + 0.14 * l);
  ctx.fill();
  const a = state.phase * 0.6;
  const gradient = ctx.createLinearGradient(
    cx + Math.cos(a) * r, cy + Math.sin(a) * r, cx - Math.cos(a) * r, cy - Math.sin(a) * r);
  colors.forEach((c, i) => gradient.addColorStop(i / (colors.length - 1), c));
  blob(r, amp, 0);
  ctx.fillStyle = gradient;
  ctx.fill();
  const gx = cx - r * 0.28, gy = cy - r * 0.32;
  const gloss = ctx.createRadialGradient(gx, gy, 0, gx, gy, r * 0.45);
  gloss.addColorStop(0, "rgba(255, 255, 255, 0.5)");
  gloss.addColorStop(1, "rgba(255, 255, 255, 0)");
  ctx.beginPath();
  ctx.ellipse(gx, gy, r * 0.45, r * 0.3, 0, 0, 2 * Math.PI);
  ctx.fillStyle = gloss;
  ctx.fill();
};

// ---- Lane

export const roundRect = (ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) => {
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
};

const PITCH = 5, BAR = 2.6, INSET = 18;

// 0...1 from a peak: -45 dBFS and below is silence.
const barHeight = (peak: number) => {
  const db = 20 * Math.log10(Math.max(peak, 1e-6));
  return Math.max(0, Math.min(1, (db + 45) / 42));
};

// One lane in the box (x, y, w, h): a bar per bin, mirrored around the middle; silence
// as dots. The right edge is `t`, so the bars slide left as time passes.
export const drawLane = (
  ctx: CanvasRenderingContext2D, x: number, y: number, width: number, height: number,
  history: WaveHistory, track: Track, t: number, color: string,
) => {
  const bin = WaveHistory.binSeconds;
  const right = width - INSET, mid = y + height / 2;
  const count = Math.floor((right - INSET) / PITCH) + 2;
  const newest = Math.floor(t / bin);
  const first = newest - count + 1;
  for (let i = 0; i < count; i++) {
    const center = (first + i + 0.5) * bin;
    const px = right - ((t - center) / bin) * PITCH;
    if (px < INSET || px > right) {
      continue;
    }
    const fade = Math.min(1, (px - INSET) / 70);
    const h = barHeight(history.peak(track, first + i));
    if (h === 0) {
      ctx.beginPath();
      ctx.arc(x + px, mid, 1, 0, 2 * Math.PI);
      ctx.fillStyle = rgba(Theme.line, fade);
      ctx.fill();
    } else {
      const bh = Math.max(BAR, h * (height - 26));
      roundRect(ctx, x + px - BAR / 2, mid - bh / 2, BAR, bh, BAR / 2);
      ctx.fillStyle = rgba(color, fade);
      ctx.fill();
    }
  }
};
