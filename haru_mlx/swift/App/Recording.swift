// Records a conversation as it was heard and exports it:
// the audio (you on the left, the model on the right) and a 1080x1080 video laid
// out like the conversation screen: the orbs and waveform lanes, the transcript below.

import AVFoundation
import SwiftUI

/// Collects both voices on the audio threads and moves them to 24 kHz 16-bit
/// storage on a background queue. Your voice starts at the first input
/// sample's ADC time; the model's is placed by the DAC time of what was played.
final class ConversationRecorder {
    private final class Pending {
        let capacity = 96_000  // 2 s at 48 kHz; drained every 100 ms
        let buffer: UnsafeMutablePointer<Float>
        var count = 0

        init() { buffer = .allocate(capacity: capacity) }
        deinit { buffer.deallocate() }

        func append(_ x: UnsafeBufferPointer<Float>) {
            let n = min(x.count, capacity - count)
            guard n > 0 else { return }
            (buffer + count).update(from: x.baseAddress!, count: n)
            count += n
        }

        func take() -> [Float] {
            defer { count = 0 }
            return Array(UnsafeBufferPointer(start: buffer, count: count))
        }
    }

    private var lock = os_unfair_lock()
    private let pendingUser = Pending()
    private let pendingModel = Pending()
    private var t0: Double?
    private var modelOffset: Int?  // 48 kHz samples from t0 to the first model sample
    private var texts: [(time: Double, text: String)] = []
    // Drain queue only
    private var user: [Int16] = []
    private var model: [Int16] = []
    private var carry: [Float] = [], modelCarry: [Float] = []
    private let queue = DispatchQueue(label: "recorder", qos: .utility)
    private let timer: DispatchSourceTimer

    init() {
        timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + 0.1, repeating: 0.1)
        timer.setEventHandler { [weak self] in self?.drain() }
        timer.resume()
    }

    /// Audio thread: input sent to the server, `adc` is its first sample's ADC time.
    func input(_ x: UnsafeBufferPointer<Float>, adc: Double) {
        os_unfair_lock_lock(&lock)
        if t0 == nil { t0 = adc }
        pendingUser.append(x)
        os_unfair_lock_unlock(&lock)
    }

    /// Audio thread: what was played, `dac` is its first sample's DAC time.
    func output(_ x: UnsafeBufferPointer<Float>, dac: Double) {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        guard let t0 else { return }
        var samples = x
        if modelOffset == nil {
            let position = Int(((dac - t0) * 48_000).rounded())
            if position + x.count <= 0 { return }
            if position < 0 { samples = UnsafeBufferPointer(rebasing: x[(-position)...]) }
            modelOffset = max(0, position)
        }
        pendingModel.append(samples)
    }

    /// Text that will be heard at host time `heardAt`.
    func text(_ s: String, heardAt: Double) {
        os_unfair_lock_lock(&lock)
        if let t0 { texts.append((heardAt - t0, s)) }
        os_unfair_lock_unlock(&lock)
    }

    private func drain() {
        os_unfair_lock_lock(&lock)
        let u = pendingUser.take(), m = pendingModel.take(), offset = modelOffset
        os_unfair_lock_unlock(&lock)
        if model.isEmpty, !m.isEmpty, let offset {
            model = [Int16](repeating: 0, count: offset / 2)
        }
        Self.decimate(u, carry: &carry, into: &user)
        Self.decimate(m, carry: &modelCarry, into: &model)
    }

    /// 48 kHz float to 24 kHz 16-bit, averaging pairs.
    private static func decimate(_ x: [Float], carry: inout [Float], into out: inout [Int16]) {
        let all = carry + x
        let pairs = all.count / 2
        out.reserveCapacity(out.count + pairs)
        for i in 0..<pairs {
            let v = max(-1, min(1, (all[2 * i] + all[2 * i + 1]) / 2))
            out.append(Int16(v * 32767))
        }
        carry = all.count % 2 == 1 ? [all.last!] : []
    }

    func finish() -> Recording {
        timer.cancel()
        return queue.sync {
            drain()
            os_unfair_lock_lock(&lock)
            let t = texts
            os_unfair_lock_unlock(&lock)
            return Recording(user: user, model: model, texts: t)
        }
    }
}

struct Recording {
    static let rate = 24_000
    let user: [Int16]
    let model: [Int16]
    let texts: [(time: Double, text: String)]

    var frames: Int { max(user.count, model.count) }
    var duration: Double { Double(frames) / Double(Self.rate) }

    private func sample(_ x: [Int16], _ i: Int) -> Int16 { x.indices.contains(i) ? x[i] : 0 }

    /// Interleaved stereo 16-bit samples [start, start + count): you left, the model right.
    func interleaved(_ start: Int, _ count: Int) -> [Int16] {
        var out = [Int16](repeating: 0, count: count * 2)
        for i in 0..<count {
            out[2 * i] = sample(user, start + i)
            out[2 * i + 1] = sample(model, start + i)
        }
        return out
    }

    func writeAudio(to url: URL) throws {
        let pcm = interleaved(0, frames)
        var data = Data()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
        let bytes = UInt32(pcm.count * 2)
        data.append(contentsOf: Array("RIFF".utf8)); u32(36 + bytes)
        data.append(contentsOf: Array("WAVEfmt ".utf8)); u32(16); u16(1); u16(2)
        u32(UInt32(Self.rate)); u32(UInt32(Self.rate * 4)); u16(4); u16(16)
        data.append(contentsOf: Array("data".utf8)); u32(bytes)
        pcm.withUnsafeBytes { data.append(contentsOf: $0) }
        try data.write(to: url)
    }

    /// The pieces of transcript heard by time `t`.
    func heardTexts(at t: Double) -> Int { texts.firstIndex { $0.time > t } ?? texts.count }

    /// Peaks per waveform bin (`WaveHistory.binSeconds`), from the start.
    private func binPeaks(_ x: [Int16]) -> [Float] {
        let n = Int(Double(Self.rate) * WaveHistory.binSeconds)
        return stride(from: 0, to: x.count, by: n).map { start in
            var peak: Int32 = 0
            for v in x[start..<min(x.count, start + n)] { peak = max(peak, abs(Int32(v))) }
            return Float(peak) / 32768
        }
    }

    private func rms(_ x: [Int16], end: Int, count: Int) -> Float {
        let start = max(0, end - count)
        guard end > start else { return 0 }
        var sum: Float = 0
        for i in start..<end {
            let v = Float(sample(x, i)) / 32768
            sum += v * v
        }
        return (sum / Float(end - start)).squareRoot()
    }

    /// H.264 + AAC mp4 of the lanes and the transcript, 1080x1080 at 30 fps: the
    /// 720-point layout drawn at 1.5x, written at its own size (no resampling).
    @MainActor
    func writeVideo(to url: URL, userName: String, modelName: String,
                    progress: @escaping (Double) -> Void) async throws {
        let layout: CGFloat = 720, scale: CGFloat = 1.5, fps = 30
        let width = Int(layout * scale), height = Int(layout * scale)
        try? FileManager.default.removeItem(at: url)
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let video = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height,
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 12_000_000,
                                              AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel],
        ])
        video.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: video, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height,
        ])
        let audio = AVAssetWriterInput(mediaType: .audio, outputSettings: [
            AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: Self.rate,
            AVNumberOfChannelsKey: 2, AVEncoderBitRateKey: 128_000,
        ])
        audio.expectsMediaDataInRealTime = false
        writer.add(video)
        writer.add(audio)
        guard writer.startWriting() else { throw writer.error ?? AudioError(description: "video writer") }
        writer.startSession(atSourceTime: .zero)

        var asbd = AudioStreamBasicDescription(
            mSampleRate: Double(Self.rate), mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kLinearPCMFormatFlagIsSignedInteger | kLinearPCMFormatFlagIsPacked,
            mBytesPerPacket: 4, mFramesPerPacket: 1, mBytesPerFrame: 4, mChannelsPerFrame: 2,
            mBitsPerChannel: 16, mReserved: 0)
        var format: CMAudioFormatDescription?
        CMAudioFormatDescriptionCreate(allocator: nil, asbd: &asbd, layoutSize: 0, layout: nil,
                                       magicCookieSize: 0, magicCookie: nil, extensions: nil,
                                       formatDescriptionOut: &format)

        let userBlob = BlobState(), modelBlob = BlobState(), logo = BlobState()
        let userPeaks = binPeaks(user), modelPeaks = binPeaks(model)
        func peaks(_ bins: [Float]) -> (Int, Int) -> [Float] {
            { first, count in (first..<first + count).map { bins.indices.contains($0) ? bins[$0] : 0 } }
        }
        let lines = TranscriptLines()
        let perFrame = Self.rate / fps
        let count = Int((duration * Double(fps)).rounded(.up))
        var modelLevel: Float = 0, userLevel: Float = 0
        let release = Float(exp(-1 / Double(fps) / 0.08))
        var audioWritten = 0
        // The writer interleaves the tracks: keep the audio a second ahead of the
        // video, and mark it finished once complete, or the writer stops taking
        // video while it waits for audio.
        func feedAudio(until limit: Int) {
            guard let format, audioWritten < frames else { return }
            while audioWritten < min(frames, limit), audio.isReadyForMoreMediaData,
                  let chunk = audioBuffer(start: audioWritten, count: 2400, format: format) {
                audio.append(chunk)
                audioWritten += 2400
            }
            if audioWritten >= frames { audio.markAsFinished() }
        }
        for k in 0..<count {
            let t = Double(k) / Double(fps)
            let end = (k + 1) * perFrame
            modelLevel = max(rms(model, end: end, count: perFrame), modelLevel * release)
            userLevel = max(rms(user, end: end, count: perFrame), userLevel * release)
            let heard = texts[..<heardTexts(at: t)]
            let shown = lines.update(heard.map(\.text))
            let recent = heard.filter { $0.time > t - Subtitles.recentSeconds }.map(\.text).joined()
            let frame = VideoFrame(
                time: t, userName: userName, modelName: modelName, logo: logo,
                user: LaneRow(name: userName, level: Double(Levels.normalized(userLevel)), time: t, blob: userBlob,
                              colors: Palette.user, barColor: Palette.userBars, peaks: peaks(userPeaks)),
                model: LaneRow(name: modelName, level: Double(Levels.normalized(modelLevel)), time: t, blob: modelBlob,
                               colors: Palette.model, barColor: Palette.modelBars, peaks: peaks(modelPeaks)),
                transcript: TranscriptLines.split(shown, recent: recent))
            let renderer = ImageRenderer(content: frame.frame(width: layout, height: layout))
            renderer.scale = scale
            guard let image = renderer.cgImage, let pool = adaptor.pixelBufferPool else { continue }
            var buffer: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer)
            guard let buffer else { continue }
            CVPixelBufferLockBaseAddress(buffer, [])
            let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buffer), width: width, height: height,
                                bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                    | CGBitmapInfo.byteOrder32Little.rawValue)
            ctx?.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            CVPixelBufferUnlockBaseAddress(buffer, [])
            feedAudio(until: end + Self.rate)
            while !video.isReadyForMoreMediaData {
                try await Task.sleep(nanoseconds: 1_000_000)
                feedAudio(until: end + Self.rate)
            }
            adaptor.append(buffer, withPresentationTime: CMTime(value: CMTimeValue(k), timescale: CMTimeScale(fps)))
            if k % 15 == 0 {
                progress(Double(k) / Double(count))
                await Task.yield()
            }
        }
        video.markAsFinished()
        while audioWritten < frames {
            feedAudio(until: frames)
            try await Task.sleep(nanoseconds: 1_000_000)
        }
        await writer.finishWriting()
        if writer.status != .completed { throw writer.error ?? AudioError(description: "video writer") }
        progress(1)
    }

    private func audioBuffer(start: Int, count: Int, format: CMAudioFormatDescription) -> CMSampleBuffer? {
        let pcm = interleaved(start, count)
        let bytes = pcm.count * 2
        var block: CMBlockBuffer?
        guard CMBlockBufferCreateWithMemoryBlock(
            allocator: nil, memoryBlock: nil, blockLength: bytes, blockAllocator: nil, customBlockSource: nil,
            offsetToData: 0, dataLength: bytes, flags: kCMBlockBufferAssureMemoryNowFlag,
            blockBufferOut: &block) == noErr, let block else { return nil }
        pcm.withUnsafeBytes { _ = CMBlockBufferReplaceDataBytes(with: $0.baseAddress!, blockBuffer: block,
                                                                offsetIntoDestination: 0, dataLength: bytes) }
        var buffer: CMSampleBuffer?
        CMAudioSampleBufferCreateReadyWithPacketDescriptions(
            allocator: nil, dataBuffer: block, formatDescription: format, sampleCount: count,
            presentationTimeStamp: CMTime(value: CMTimeValue(start), timescale: CMTimeScale(Self.rate)),
            packetDescriptions: nil, sampleBufferOut: &buffer)
        return buffer
    }
}

/// One video frame, laid out like the conversation screen: the name on top,
/// your lane and the model's, the transcript below (the last seconds dark).
struct VideoFrame: View {
    let time: Double
    let userName: String
    let modelName: String
    let logo: BlobState
    let user: LaneRow
    let model: LaneRow
    let transcript: (older: String, recent: String)

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 6) {
                Canvas { ctx, size in
                    BlobPainter.draw(ctx, size: size, level: 0, time: time, state: logo, colors: Palette.model)
                }
                .frame(width: 34, height: 34)
                Text("Haru").font(.system(size: 26, weight: .semibold)).tracking(-0.3)
            }
            .padding(.bottom, 8)
            user
            model
            (Text(transcript.older).foregroundColor(Theme.muted) + Text(transcript.recent).foregroundColor(Theme.ink))
                .font(.system(size: TranscriptLines.fontSize, weight: .medium))
                .lineSpacing(TranscriptLines.lineSpacing)
                .fixedSize(horizontal: false, vertical: true)
                .frame(width: TranscriptLines.width, height: TranscriptLines.height, alignment: .topLeading)
                .clipped()
                .padding(.leading, 126)
                .padding(.top, 10)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 32)
        .padding(.vertical, 32)
        .foregroundStyle(Theme.ink)
        .background(Theme.paper)
        .environment(\.colorScheme, .light)
    }
}

/// The video's transcript: wrapped into lines from the start and shown from
/// the top; once the box is full, a new line scrolls the whole top line away.
final class TranscriptLines {
    static let width: CGFloat = 530, height: CGFloat = 300, fontSize: CGFloat = 26, lineSpacing: CGFloat = 8
    private let font = NSFont.systemFont(ofSize: fontSize, weight: .medium)
    private lazy var visible: Int = {
        let line = NSLayoutManager().defaultLineHeight(for: font)
        return max(1, Int((Self.height + Self.lineSpacing) / (line + Self.lineSpacing)))
    }()
    private var heard = 0    // pieces taken so far
    private var shown = ""   // text from the first visible line on
    private var rendered = ""

    /// The visible lines, joined by newlines, for all pieces heard so far.
    func update(_ pieces: [String]) -> String {
        guard pieces.count != heard else { return rendered }
        shown += pieces[heard...].joined()
        heard = pieces.count
        var lines = wrap(shown)
        if lines.count > visible {
            lines = Array(lines.suffix(visible))
            shown = lines.joined()
        }
        rendered = lines.map { $0.trimmingCharacters(in: .newlines) }.joined(separator: "\n")
        return rendered
    }

    /// `shown` split where its last `recent` characters begin (newlines,
    /// which `update` adds between lines, aside).
    static func split(_ shown: String, recent: String) -> (older: String, recent: String) {
        var left = recent.filter { !$0.isNewline }.count
        var index = shown.endIndex
        while left > 0, index > shown.startIndex {
            index = shown.index(before: index)
            if !shown[index].isNewline { left -= 1 }
        }
        return (String(shown[..<index]), String(shown[index...]))
    }

    /// Lines as laid out a little narrower than the box, so that SwiftUI never
    /// wraps one of them again.
    private func wrap(_ text: String) -> [String] {
        let storage = NSTextStorage(string: text, attributes: [.font: font])
        let layout = NSLayoutManager()
        let container = NSTextContainer(size: CGSize(width: Self.width - 12, height: .greatestFiniteMagnitude))
        container.lineFragmentPadding = 0
        layout.addTextContainer(container)
        storage.addLayoutManager(layout)
        let ns = text as NSString
        var lines: [String] = []
        layout.enumerateLineFragments(forGlyphRange: layout.glyphRange(for: container)) { _, _, _, glyphs, _ in
            lines.append(ns.substring(with: layout.characterRange(forGlyphRange: glyphs, actualGlyphRange: nil)))
        }
        return lines
    }
}
