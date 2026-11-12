// The conversation as two waveform lanes on one clock: your voice and the
// model's, newest at the right, each with a living orb on the left, and the
// model's words below as they are heard.

import SwiftUI

// MARK: - History

/// Loudness per lane on the host clock, 20 bins a second, filled on the audio
/// threads at the time each sample is heard: your voice at its ADC time, the
/// model's at its DAC time. The model's text is kept with the time it is heard.
final class WaveHistory {
    enum Track: Int { case user, model }

    static let binSeconds = 0.05
    private static let capacity = 1200  // one minute

    private var lock = os_unfair_lock()
    // Fixed buffers, so the audio threads never allocate.
    private let peaks = [UnsafeMutablePointer<Float>.allocate(capacity: capacity),
                         UnsafeMutablePointer<Float>.allocate(capacity: capacity)]
    private var last = [Int.min, Int.min]  // newest bin written per lane
    private var texts: [(time: Double, text: String)] = []
    private var settled = 0, settledText = ""  // texts older than the recent window

    init() { reset() }

    deinit { peaks.forEach { $0.deallocate() } }

    func reset() {
        os_unfair_lock_lock(&lock)
        for p in peaks { p.update(repeating: 0, count: Self.capacity) }
        last = [Int.min, Int.min]
        texts.removeAll()
        settled = 0
        settledText = ""
        os_unfair_lock_unlock(&lock)
    }

    private static func slot(_ bin: Int) -> Int { ((bin % capacity) + capacity) % capacity }

    /// Audio thread: input sent to the server, `time` is its first sample's ADC time.
    func input(_ x: UnsafeBufferPointer<Float>, at time: Double) { add(x, time, .user) }

    /// Audio thread: what was played, `time` is its first sample's DAC time.
    func output(_ x: UnsafeBufferPointer<Float>, at time: Double) { add(x, time, .model) }

    private func add(_ x: UnsafeBufferPointer<Float>, _ time: Double, _ track: Track) {
        var peak: Float = 0
        for v in x { peak = max(peak, abs(v)) }
        let bin = Int((time / Self.binSeconds).rounded(.down))
        let lane = track.rawValue, p = peaks[lane]
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        let newest = last[lane]
        if newest == Int.min || bin - newest >= Self.capacity {
            p.update(repeating: 0, count: Self.capacity)
            last[lane] = bin
        } else if bin > newest {
            for b in (newest + 1)...bin { p[Self.slot(b)] = 0 }
            last[lane] = bin
        } else if newest - bin >= Self.capacity {
            return
        }
        p[Self.slot(bin)] = max(p[Self.slot(bin)], peak)
    }

    /// Peaks of `count` bins from `first` (0 where nothing was heard).
    func peaks(_ track: Track, from first: Int, count: Int) -> [Float] {
        var out = [Float](repeating: 0, count: count)
        os_unfair_lock_lock(&lock)
        let newest = last[track.rawValue], p = peaks[track.rawValue]
        for i in 0..<count {
            let b = first + i
            if newest != Int.min, b <= newest, newest - b < Self.capacity { out[i] = p[Self.slot(b)] }
        }
        os_unfair_lock_unlock(&lock)
        return out
    }

    /// Text that will be heard at host time `heardAt`.
    func text(_ s: String, heardAt: Double) {
        os_unfair_lock_lock(&lock)
        texts.append((heardAt, s))
        os_unfair_lock_unlock(&lock)
    }

    /// The model's text heard by `time`: what is older than `recent` seconds,
    /// and the rest.
    func heard(at time: Double, recent: Double) -> (older: String, recent: String) {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        while settled < texts.count, texts[settled].time <= time - recent {
            settledText += texts[settled].text
            settled += 1
        }
        var latest = ""
        for entry in texts[settled...] where entry.time <= time { latest += entry.text }
        return (settledText, latest)
    }
}

// MARK: - Palette

/// Warm: you in a yellowish jade green, the model in coral to amber.
enum Palette {
    static let user = [Color(hex: 0xa3d977), Color(hex: 0x3fb984), Color(hex: 0xe8d36a)]
    static let model = [Color(hex: 0xff9a7b), Color(hex: 0xf9614f), Color(hex: 0xffb35c)]
    static let userBars = Color(hex: 0x3aa877)
    static let modelBars = Color(hex: 0xf26b4f)
    static let lane = Color(hex: 0xf4f5f7)
}

// MARK: - Orb

/// Smoothing factor for a time constant, independent of the frame rate.
func smoothing(_ dt: Double, _ tau: Double) -> Double { 1 - exp(-dt / tau) }

final class BlobState {
    var level = 0.0
    var phase = Double.random(in: 0..<10)
    var lastTime: Double?
}

/// A soft, always-moving blob: at rest it breathes, ripples and turns on its
/// own; with the voice it swells, wobbles harder and turns faster.
enum BlobPainter {
    static func draw(_ ctx: GraphicsContext, size: CGSize, level raw: Double, time t: Double,
                     state: BlobState, colors: [Color]) {
        let dt = min(0.1, state.lastTime.map { t - $0 } ?? 0)
        state.lastTime = t
        // Background noise (below ~-48 dBFS) leaves it at rest.
        let target = max(0, min(1, (raw - 0.2) / 0.6))
        state.level += (target - state.level) * smoothing(dt, target > state.level ? 0.05 : 0.25)
        let l = state.level
        state.phase += dt * (1.7 + 2.2 * l)

        let w = min(size.width, size.height)
        let c = CGPoint(x: size.width / 2, y: size.height / 2)
        // Nearly a circle: the motion is mostly breathing and turning, the
        // wobble slight. Breathing fades out as the voice takes over. The outer
        // layer fits at either end: 0.22 * 1.06 * 1.2 * (1 + 1.2 * 0.025) and
        // 0.22 * 1.4 * 1.2 * (1 + 1.2 * 0.07) < 0.5.
        let breath = 1 + 0.06 * sin(t * 1.6) * (1 - l)
        let r = w * 0.22 * (1 + 0.4 * l) * breath
        let amp = 0.025 + 0.045 * l

        func blob(_ radius: Double, _ amp: Double, _ shift: Double) -> Path {
            var path = Path()
            let n = 96, ph = state.phase + shift
            for i in 0...n {
                let th = Double(i) / Double(n) * 2 * .pi
                let k = 1 + amp * (0.6 * sin(2 * th + 0.7 * ph + 4.1) + 0.45 * sin(3 * th + 1.3 * ph)
                                   + 0.15 * sin(5 * th - 1.1 * ph + 1.7))
                let point = CGPoint(x: c.x + radius * k * cos(th), y: c.y + radius * k * sin(th))
                if i == 0 { path.move(to: point) } else { path.addLine(to: point) }
            }
            path.closeSubpath()
            return path
        }

        ctx.fill(blob(r * 1.2, amp * 1.2, 2.1), with: .color(colors[2].opacity(0.14 + 0.16 * l)))
        ctx.fill(blob(r * 1.08, amp * 1.1, 1.0), with: .color(colors[0].opacity(0.26 + 0.14 * l)))
        let a = state.phase * 0.6
        ctx.fill(blob(r, amp, 0), with: .linearGradient(
            Gradient(colors: colors),
            startPoint: CGPoint(x: c.x + cos(a) * r, y: c.y + sin(a) * r),
            endPoint: CGPoint(x: c.x - cos(a) * r, y: c.y - sin(a) * r)))
        let gloss = CGPoint(x: c.x - r * 0.28, y: c.y - r * 0.32)
        ctx.fill(Path(ellipseIn: CGRect(x: gloss.x - r * 0.45, y: gloss.y - r * 0.3, width: r * 0.9, height: r * 0.6)),
                 with: .radialGradient(Gradient(colors: [.white.opacity(0.5), .white.opacity(0)]),
                                       center: gloss, startRadius: 0, endRadius: r * 0.45))
    }
}

// MARK: - Lane

/// One lane: a bar per bin, mirrored around the middle; silence as dots.
/// The right edge is `now`, so the bars slide left as time passes.
enum WaveLanePainter {
    static let pitch = 5.0, bar = 2.6, inset = 18.0

    /// 0...1 from a peak: -45 dBFS and below is silence.
    static func height(_ peak: Float) -> Double {
        let db = 20 * log10(Double(max(peak, 1e-6)))
        return max(0, min(1, (db + 45) / 42))
    }

    /// `peaks(first, count)` gives the peaks of bins `first..<first + count`.
    static func draw(_ ctx: GraphicsContext, size: CGSize, peaks: (Int, Int) -> [Float],
                     now t: Double, color: Color) {
        let right = size.width - inset, mid = size.height / 2
        let count = Int((right - inset) / pitch) + 2
        let newest = Int((t / WaveHistory.binSeconds).rounded(.down))
        let first = newest - count + 1
        let values = peaks(first, count)
        for (i, peak) in values.enumerated() {
            let center = (Double(first + i) + 0.5) * WaveHistory.binSeconds
            let x = right - (t - center) / WaveHistory.binSeconds * pitch
            guard x >= inset, x <= right else { continue }
            let fade = min(1, (x - inset) / 70)
            let h = height(peak)
            if h == 0 {
                ctx.fill(Path(ellipseIn: CGRect(x: x - 1, y: mid - 1, width: 2, height: 2)),
                         with: .color(Theme.line.opacity(fade)))
            } else {
                let barHeight = max(bar, h * (size.height - 26))
                ctx.fill(Path(roundedRect: CGRect(x: x - bar / 2, y: mid - barHeight / 2, width: bar, height: barHeight),
                              cornerRadius: bar / 2),
                         with: .color(color.opacity(fade)))
            }
        }
    }
}

// MARK: - Views

/// A lane with its orb and name on the left, as in the app and the video.
struct LaneRow: View {
    let name: String
    let level: Double
    let time: Double
    let blob: BlobState
    let colors: [Color]
    let barColor: Color
    let peaks: (Int, Int) -> [Float]
    var date: Date? = nil  // the live screen frame: Canvas redraws only when what it reads changes

    var body: some View {
        HStack(spacing: 16) {
            VStack(spacing: 4) {
                Canvas { ctx, size in
                    _ = date
                    BlobPainter.draw(ctx, size: size, level: level, time: time, state: blob, colors: colors)
                }
                .frame(width: 92, height: 92)
                Text(name).font(.system(size: 12, weight: .medium)).foregroundStyle(Theme.muted).lineLimit(1)
            }
            .frame(width: 110)
            Canvas { ctx, size in
                _ = date
                WaveLanePainter.draw(ctx, size: size, peaks: peaks, now: time, color: barColor)
            }
            .frame(height: 112)
            .background(RoundedRectangle(cornerRadius: 22, style: .continuous).fill(Palette.lane))
        }
    }
}

/// Both lanes, drawn from one clock reading per screen frame so they line up.
struct WaveLanes: View {
    let levels: Levels
    let waves: WaveHistory
    let userName: String
    let modelName: String
    @State private var userBlob = BlobState()
    @State private var modelBlob = BlobState()

    var body: some View {
        TimelineView(.animation) { timeline in
            let t = now()  // the host clock the bins use
            let level = levels.current()
            VStack(spacing: 14) {
                LaneRow(name: userName, level: Double(level.user), time: t, blob: userBlob, colors: Palette.user,
                        barColor: Palette.userBars, peaks: { waves.peaks(.user, from: $0, count: $1) },
                        date: timeline.date)
                LaneRow(name: modelName, level: Double(level.model), time: t, blob: modelBlob, colors: Palette.model,
                        barColor: Palette.modelBars, peaks: { waves.peaks(.model, from: $0, count: $1) },
                        date: timeline.date)
            }
        }
    }
}

/// The model's words as they are heard: the last few seconds dark, the rest grey.
struct Subtitles: View {
    static let recentSeconds = 2.5
    let waves: WaveHistory

    var body: some View {
        TimelineView(.periodic(from: .now, by: 0.05)) { _ in
            let heard = waves.heard(at: now(), recent: Self.recentSeconds)
            SubtitleText(older: heard.older, recent: heard.recent).equatable()
        }
    }
}

private struct SubtitleText: View, Equatable {
    let older: String
    let recent: String

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                Group {
                    if older.isEmpty && recent.isEmpty {
                        Text("…").foregroundColor(Theme.muted)
                    } else {
                        Text(older).foregroundColor(Theme.muted) + Text(recent).foregroundColor(Theme.ink)
                    }
                }
                .font(.system(size: 22, weight: .medium))
                .lineSpacing(7)
                .frame(maxWidth: .infinity, alignment: .leading)
                .textSelection(.enabled)
                .padding(.horizontal, 4)
                Color.clear.frame(height: 1).id("end")
            }
            .onChange(of: older.count + recent.count) { _ in proxy.scrollTo("end", anchor: .bottom) }
        }
    }
}
