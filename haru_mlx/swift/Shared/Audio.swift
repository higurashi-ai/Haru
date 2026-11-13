// Audio I/O for the full-duplex client: Core Audio HAL output units with small
// device buffers, the playback schedule, and the reply-latency measurement.

import AudioToolbox
import CoreAudio
import Darwin
import Foundation

// MARK: - Clock

let timebase: mach_timebase_info_data_t = {
    var info = mach_timebase_info_data_t()
    mach_timebase_info(&info)
    return info
}()

func hostSeconds(_ host: UInt64) -> Double {
    Double(host) * Double(timebase.numer) / Double(timebase.denom) / 1e9
}

func now() -> Double { hostSeconds(mach_absolute_time()) }

// MARK: - Devices

struct AudioError: Error, CustomStringConvertible {
    let description: String
}

func ok(_ status: OSStatus, _ what: String) throws {
    if status != noErr { throw AudioError(description: "\(what) failed (\(status))") }
}

func property<T>(_ id: AudioObjectID, _ selector: AudioObjectPropertySelector,
                 _ scope: AudioObjectPropertyScope, _ value: T) -> T {
    var address = AudioObjectPropertyAddress(
        mSelector: selector, mScope: scope, mElement: kAudioObjectPropertyElementMain)
    var result = value
    var size = UInt32(MemoryLayout<T>.size)
    AudioObjectGetPropertyData(id, &address, 0, nil, &size, &result)
    return result
}

struct Device: Hashable {
    let id: AudioDeviceID
    let name: String
    let inputs: Int
    let outputs: Int
    let rate: Double
}

private func channels(_ id: AudioDeviceID, _ scope: AudioObjectPropertyScope) -> Int {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyStreamConfiguration, mScope: scope,
        mElement: kAudioObjectPropertyElementMain)
    var size: UInt32 = 0
    AudioObjectGetPropertyDataSize(id, &address, 0, nil, &size)
    let list = UnsafeMutableRawPointer.allocate(byteCount: Int(max(size, 16)), alignment: 16)
    defer { list.deallocate() }
    AudioObjectGetPropertyData(id, &address, 0, nil, &size, list)
    let buffers = UnsafeMutableAudioBufferListPointer(
        list.assumingMemoryBound(to: AudioBufferList.self))
    return buffers.reduce(0) { $0 + Int($1.mNumberChannels) }
}

private func deviceName(_ id: AudioDeviceID) -> String {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioObjectPropertyName, mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var name: Unmanaged<CFString>?
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    AudioObjectGetPropertyData(id, &address, 0, nil, &size, &name)
    return name?.takeRetainedValue() as String? ?? "?"
}

func audioDevices() -> [Device] {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDevices, mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    let system = AudioObjectID(kAudioObjectSystemObject)
    var size: UInt32 = 0
    AudioObjectGetPropertyDataSize(system, &address, 0, nil, &size)
    var ids = [AudioDeviceID](repeating: 0, count: Int(size) / MemoryLayout<AudioDeviceID>.size)
    AudioObjectGetPropertyData(system, &address, 0, nil, &size, &ids)
    return ids.map { id in
        Device(id: id, name: deviceName(id),
               inputs: channels(id, kAudioObjectPropertyScopeInput),
               outputs: channels(id, kAudioObjectPropertyScopeOutput),
               rate: property(id, kAudioDevicePropertyNominalSampleRate,
                              kAudioObjectPropertyScopeGlobal, Float64(0)))
    }
}

func defaultDevice(input: Bool) -> AudioDeviceID {
    property(AudioObjectID(kAudioObjectSystemObject),
             input ? kAudioHardwarePropertyDefaultInputDevice : kAudioHardwarePropertyDefaultOutputDevice,
             kAudioObjectPropertyScopeGlobal, AudioDeviceID(0))
}

/// The named device (or the system default) that has inputs / outputs.
func findDevice(_ name: String?, input: Bool) -> Device? {
    let id = defaultDevice(input: input)
    return audioDevices().first { d in
        (name == nil || name!.isEmpty ? d.id == id : d.name == name)
            && (input ? d.inputs > 0 : d.outputs > 0)
    }
}

/// Device latency + safety offset + stream latency, in frames, as the driver states them.
private func deviceLatencyFrames(_ id: AudioDeviceID, input: Bool) -> UInt32 {
    let scope = input ? kAudioObjectPropertyScopeInput : kAudioObjectPropertyScopeOutput
    let latency: UInt32 = property(id, kAudioDevicePropertyLatency, scope, UInt32(0))
    let safety: UInt32 = property(id, kAudioDevicePropertySafetyOffset, scope, UInt32(0))
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyStreams, mScope: scope,
        mElement: kAudioObjectPropertyElementMain)
    var size: UInt32 = 0
    AudioObjectGetPropertyDataSize(id, &address, 0, nil, &size)
    var streams = [AudioStreamID](repeating: 0, count: Int(size) / MemoryLayout<AudioStreamID>.size)
    AudioObjectGetPropertyData(id, &address, 0, nil, &size, &streams)
    let stream: UInt32 = streams.first.map {
        property($0, kAudioStreamPropertyLatency, kAudioObjectPropertyScopeGlobal, UInt32(0))
    } ?? 0
    return latency + safety + stream
}

/// Ask for `frames` per IO cycle and return what the device actually uses
/// (Bluetooth devices, for one, keep larger buffers).
private func setBufferFrames(_ id: AudioDeviceID, _ frames: UInt32) -> UInt32 {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyBufferFrameSize, mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var value = frames
    AudioObjectSetPropertyData(id, &address, 0, nil, UInt32(MemoryLayout<UInt32>.size), &value)
    return property(id, kAudioDevicePropertyBufferFrameSize, kAudioObjectPropertyScopeGlobal, frames)
}

// MARK: - Catch-up

/// Shorten `x` by `rate` (0..0.25) without changing its pitch (WSOLA): Hann
/// windows at 50 % overlap, each taken near its nominal (faster) input position
/// where it best continues the previous one. The first half window and the
/// last half window keep the input's own samples, so frames still join.
func wsolaCompress(_ x: [Float], rate: Double, win: Int = 960, tol: Int = 240) -> [Float] {
    let n = x.count
    let outLen = Int((Double(n) * (1 - rate)).rounded())
    let hop = win / 2
    if rate <= 0 || n < 4 * win || outLen <= win { return x }
    let speed = Double(n) / Double(outLen)
    let w = (0..<win).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(win))) }
    var out = [Float](repeating: 0, count: outLen + win)
    var wsum = [Float](repeating: 0, count: outLen + win)
    for i in 0..<win {
        let wi = i < hop ? 1 : w[i]
        out[i] += x[i] * wi
        wsum[i] += wi
    }
    var prev = 0
    var k = 1
    while k * hop + hop <= outLen {
        let nominal = Int((Double(k * hop) * speed).rounded())
        let natural = prev + hop  // what would follow the previous window unchanged
        let hi = min(n - win, nominal + tol)
        let lo = min(max(0, nominal - tol), hi)
        var best = (lo...hi).contains(nominal) ? nominal : lo
        if natural + win <= n {
            var bestScore = -Float.infinity
            for start in lo...hi {
                var score: Float = 0
                for i in stride(from: 0, to: win, by: 4) { score += x[start + i] * x[natural + i] }
                if score > bestScore {
                    bestScore = score
                    best = start
                }
            }
        }
        for i in 0..<win {
            out[k * hop + i] += x[best + i] * w[i]
            wsum[k * hop + i] += w[i]
        }
        prev = best
        k += 1
    }
    var y = (0..<outLen).map { wsum[$0] > 1e-3 ? out[$0] / wsum[$0] : 0 }
    for i in 0..<hop {  // crossfade into the input's tail
        let r = Float(i + 1) / Float(hop)
        y[outLen - hop + i] = y[outLen - hop + i] * (1 - r) + x[n - hop + i] * r
    }
    return y
}

// MARK: - Playback

/// Model frame 0 starts playing at `startAt` (the first input sample's ADC
/// time plus the time model frame 1 needs), so frame 1 is ready when frame 0
/// ends whatever order the devices started in. After an underrun, `margin` of
/// silence restores the slack. Catch-up: while more than `margin` + 5 ms is
/// queued when a frame arrives (after an underrun, or frames that came late
/// together), the frame is shortened by up to 10 % (20 % above 150 ms) with
/// WSOLA until playback is back on schedule.
final class Player {
    var lock = os_unfair_lock()
    var chunks: [(Int, [Float])] = []
    var offset = 0
    var pad = 0
    var margin: Int
    let rate: Double
    var startAt: Double? = nil  // DAC time of model frame 0
    var started = false
    var dry = false
    var underruns = 0
    var compressed = 0  // samples removed by catch-up
    var dacStart: [Int: Double] = [:]
    var writtenAt: [Int: Double] = [:]
    var recent: [(Int, Double)] = []  // last frames' (index, DAC start), for live display

    init(margin: Int, rate: Double) {
        self.margin = margin
        self.rate = rate
    }

    func push(_ index: Int, _ pcm: [Float]) {
        os_unfair_lock_lock(&lock)
        let queued = chunks.reduce(0) { $0 + $1.1.count } - offset + pad
        let playing = started && !dacStart.isEmpty
        os_unfair_lock_unlock(&lock)
        var frame = pcm
        let excess = queued - margin
        if playing && excess > Int(0.005 * rate) {
            let cap = excess > Int(0.150 * rate) ? 0.20 : 0.10
            frame = wsolaCompress(pcm, rate: min(cap, Double(excess) / Double(pcm.count)))
        }
        os_unfair_lock_lock(&lock)
        compressed += pcm.count - frame.count
        chunks.append((index, frame))
        os_unfair_lock_unlock(&lock)
    }

    func pull(_ out: UnsafeMutableBufferPointer<Float>, dac: Double) {
        let t = now()
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        if !started {
            let wait = startAt.map { ($0 - dac) * rate } ?? .infinity
            if wait >= Double(out.count) {
                for i in 0..<out.count { out[i] = 0 }
                return
            }
            started = true
            pad = max(0, Int(wait))
        }
        if dry && !chunks.isEmpty {
            dry = false
            pad = margin
        }
        var written = 0
        while written < out.count {
            if pad > 0 {
                let take = min(pad, out.count - written)
                for i in 0..<take { out[written + i] = 0 }
                written += take
                pad -= take
                continue
            }
            guard !chunks.isEmpty else { break }
            let (index, pcm) = chunks[0]
            if offset == 0 {
                dacStart[index] = dac + Double(written) / rate
                writtenAt[index] = t
                recent.append((index, dacStart[index]!))
                if recent.count > 64 { recent.removeFirst() }
            }
            let take = min(pcm.count - offset, out.count - written)
            for i in 0..<take { out[written + i] = pcm[offset + i] }
            written += take
            offset += take
            if offset == pcm.count {
                chunks.removeFirst()
                offset = 0
            }
        }
        if written < out.count {
            for i in written..<out.count { out[i] = 0 }
            if !dry {
                dry = true
                underruns += 1
            }
        }
    }
}

// MARK: - Input resampling (device rate -> 48 kHz)

/// Streaming linear interpolation; identity when the rates match.
struct Resampler {
    let ratio: Double  // input samples per output sample
    var position = 0.0  // next output position, in input samples from `last`
    var last: Float = 0

    init(from input: Double, to output: Double) { ratio = input / output }

    mutating func process(_ x: UnsafeBufferPointer<Float>, into out: inout [Float]) {
        if ratio == 1 {
            out.append(contentsOf: x)
            return
        }
        while position <= Double(x.count - 1) {  // samples indexed from -1 (`last`)
            let i = Int(position.rounded(.down))
            let frac = Float(position - Double(i))
            let a = i < 0 ? last : x[i]
            let b = x[i + 1]
            out.append(a + (b - a) * frac)
            position += ratio
        }
        position -= Double(x.count)
        if let tail = x.last { last = tail }
    }
}

// MARK: - Duplex audio

struct LatencyReport {
    var frames = 0
    var p50 = 0.0, p95 = 0.0, minimum = 0.0, maximum = 0.0
    var underruns = 0
    var caughtUpMs = 0.0
    var parts: [(String, Double)] = []
}

func median(_ v: [Double]) -> Double {
    let s = v.sorted()
    return s.isEmpty ? 0 : s[s.count / 2]
}

/// Microphone in and model audio out at 48 kHz, plus the reply latency on the
/// audio clock: from the last sample of each user frame at the ADC to the
/// first sample of the model frame answering it at the DAC.
final class DuplexAudio {
    let input: Device
    let output: Device
    let rate = 48_000.0
    let frameSamples: Int
    let bufferMs: Double
    private(set) var lead: Double
    private(set) var inputBufferMs = 0.0
    private(set) var outputBufferMs = 0.0
    private(set) var marginMs: Double
    let player: Player
    let inLatency: Double
    let outLatency: Double
    private let onAudio: (UnsafeBufferPointer<Float>) -> Void
    private var inUnit: AudioUnit?
    private var outUnit: AudioUnit?
    private let buffers: UnsafeMutableAudioBufferListPointer
    private let capacity = 8192
    private var resampler: Resampler
    private var resampled: [Float] = []
    private var sent = 0  // 48 kHz samples sent so far

    // Measurements (audio and socket threads write, others read under the lock).
    private let statsLock = NSLock()
    private var adcEnd: [Int: Double] = [:]      // user frame k: ADC time of its last sample
    private var seen: [Int: Double] = [:]        // user frame k: when the input callback had it
    private var received: [Int: Double] = [:]    // model frame j: when it arrived from the server
    private var serverStep: [Int: Double] = [:]  // model frame j: step time the server sent
    private(set) var inputLevel: Float = 0
    private(set) var outputLevel: Float = 0
    /// Called on the audio threads with the 48 kHz input sent to the server
    /// (and the ADC time of its first sample), and with what is played (and the
    /// DAC time of its first sample). Set before start().
    var inputTap: ((UnsafeBufferPointer<Float>, Double) -> Void)?
    var outputTap: ((UnsafeBufferPointer<Float>, Double) -> Void)?

    init(input: Device, output: Device, bufferMs: Double, marginMs: Double, frameSamples: Int,
         stepMs: Double, onAudio: @escaping (UnsafeBufferPointer<Float>) -> Void) {
        self.input = input
        self.output = output
        self.frameSamples = frameSamples
        self.bufferMs = bufferMs
        self.marginMs = marginMs
        self.stepMs = stepMs
        self.onAudio = onAudio
        player = Player(margin: Int(marginMs / 1000 * 48_000), rate: 48_000)
        inLatency = Double(deviceLatencyFrames(input.id, input: true)) / input.rate
        outLatency = Double(deviceLatencyFrames(output.id, input: false)) / output.rate
        // From a user frame's last sample at the ADC to the answering model frame
        // at the DAC: input buffer, step, ~5 ms for the socket and encoding the
        // frame's last chunk, output buffer and device, and the margin.
        lead = bufferMs / 1000 + stepMs / 1000 + 0.005 + bufferMs / 1000 + outLatency + marginMs / 1000
        buffers = AudioBufferList.allocate(maximumBuffers: max(1, input.inputs))
        for i in 0..<max(1, input.inputs) {
            buffers[i] = AudioBuffer(mNumberChannels: 1, mDataByteSize: UInt32(capacity * 4),
                                     mData: UnsafeMutableRawPointer.allocate(byteCount: capacity * 4,
                                                                             alignment: 16))
        }
        resampler = Resampler(from: input.rate, to: 48_000)
        resampled.reserveCapacity(capacity * 2)
    }

    deinit {
        for b in buffers { b.mData?.deallocate() }
        free(buffers.unsafeMutablePointer)
    }

    private static func makeUnit(device: AudioDeviceID, input: Bool) throws -> AudioUnit {
        var desc = AudioComponentDescription(
            componentType: kAudioUnitType_Output, componentSubType: kAudioUnitSubType_HALOutput,
            componentManufacturer: kAudioUnitManufacturer_Apple, componentFlags: 0, componentFlagsMask: 0)
        var unit: AudioUnit?
        try ok(AudioComponentInstanceNew(AudioComponentFindNext(nil, &desc)!, &unit), "AudioComponentInstanceNew")
        let u = unit!
        var on: UInt32 = input ? 1 : 0, off: UInt32 = input ? 0 : 1
        try ok(AudioUnitSetProperty(u, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input, 1, &on, 4),
               "EnableIO input")
        try ok(AudioUnitSetProperty(u, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, 0, &off, 4),
               "EnableIO output")
        var id = device
        try ok(AudioUnitSetProperty(u, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0,
                                    &id, UInt32(MemoryLayout<AudioDeviceID>.size)), "CurrentDevice")
        return u
    }

    private static func floatFormat(rate: Double, channels: Int) -> AudioStreamBasicDescription {
        AudioStreamBasicDescription(
            mSampleRate: rate, mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked | kAudioFormatFlagIsNonInterleaved,
            mBytesPerPacket: 4, mFramesPerPacket: 1, mBytesPerFrame: 4,
            mChannelsPerFrame: UInt32(channels), mBitsPerChannel: 32, mReserved: 0)
    }

    private let stepMs: Double

    func start() throws {
        let inFrames = setBufferFrames(input.id, UInt32(input.rate * bufferMs / 1000))
        let outFrames = setBufferFrames(output.id, UInt32(output.rate * bufferMs / 1000))
        inputBufferMs = Double(inFrames) / input.rate * 1000
        outputBufferMs = Double(outFrames) / output.rate * 1000
        // A frame can land just after an output cycle began: keep at least one
        // output cycle (plus 5 ms) of slack, and plan with the real buffers.
        marginMs = max(marginMs, outputBufferMs + 5)
        os_unfair_lock_lock(&player.lock)
        player.margin = Int(marginMs / 1000 * rate)
        os_unfair_lock_unlock(&player.lock)
        lead = inputBufferMs / 1000 + stepMs / 1000 + 0.005 + outputBufferMs / 1000 + outLatency
            + marginMs / 1000
        let size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        let refCon = Unmanaged.passUnretained(self).toOpaque()

        let inU = try Self.makeUnit(device: input.id, input: true)
        var inFormat = Self.floatFormat(rate: input.rate, channels: max(1, input.inputs))
        try ok(AudioUnitSetProperty(inU, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 1,
                                    &inFormat, size), "input format")
        var inCallback = AURenderCallbackStruct(inputProc: { refCon, flags, ts, bus, frames, _ in
            Unmanaged<DuplexAudio>.fromOpaque(refCon).takeUnretainedValue()
                .handleInput(flags, ts, bus, frames)
        }, inputProcRefCon: refCon)
        try ok(AudioUnitSetProperty(inU, kAudioOutputUnitProperty_SetInputCallback, kAudioUnitScope_Global, 0,
                                    &inCallback, UInt32(MemoryLayout<AURenderCallbackStruct>.size)),
               "input callback")

        let outU = try Self.makeUnit(device: output.id, input: false)
        var outFormat = Self.floatFormat(rate: rate, channels: 1)  // the unit converts to the device
        try ok(AudioUnitSetProperty(outU, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0,
                                    &outFormat, size), "output format")
        var outCallback = AURenderCallbackStruct(inputProc: { refCon, _, ts, _, frames, data in
            Unmanaged<DuplexAudio>.fromOpaque(refCon).takeUnretainedValue()
                .handleOutput(ts, frames, data!)
        }, inputProcRefCon: refCon)
        try ok(AudioUnitSetProperty(outU, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0,
                                    &outCallback, UInt32(MemoryLayout<AURenderCallbackStruct>.size)),
               "render callback")

        try ok(AudioUnitInitialize(inU), "initialize input")
        try ok(AudioUnitInitialize(outU), "initialize output")
        inUnit = inU
        outUnit = outU
        try ok(AudioOutputUnitStart(outU), "start output")
        try ok(AudioOutputUnitStart(inU), "start input")
    }

    func stop() {
        if let u = outUnit { AudioOutputUnitStop(u); AudioComponentInstanceDispose(u) }
        if let u = inUnit { AudioOutputUnitStop(u); AudioComponentInstanceDispose(u) }
        outUnit = nil
        inUnit = nil
    }

    private func handleInput(_ flags: UnsafeMutablePointer<AudioUnitRenderActionFlags>,
                             _ ts: UnsafePointer<AudioTimeStamp>, _ bus: UInt32, _ frames: UInt32) -> OSStatus {
        for i in 0..<buffers.count { buffers[i].mDataByteSize = frames * 4 }
        let status = AudioUnitRender(inUnit!, flags, ts, bus, frames, buffers.unsafeMutablePointer)
        if status != noErr { return status }
        let seenAt = now()
        let adc = hostSeconds(ts.pointee.mHostTime) - inLatency  // ADC time of the first sample
        if sent == 0 {
            os_unfair_lock_lock(&player.lock)
            player.startAt = adc + lead
            os_unfair_lock_unlock(&player.lock)
        }
        let mono = UnsafeBufferPointer(start: buffers[0].mData!.assumingMemoryBound(to: Float.self),
                                       count: Int(frames))
        inputLevel = max(rms(mono), inputLevel * levelDecay(frames))
        resampled.removeAll(keepingCapacity: true)
        resampler.process(mono, into: &resampled)
        let start = sent
        sent += resampled.count
        var k = start / frameSamples
        while (k + 1) * frameSamples <= sent {
            let last = (k + 1) * frameSamples - 1 - start
            statsLock.lock()
            adcEnd[k] = adc + Double(last) / rate
            seen[k] = seenAt
            statsLock.unlock()
            k += 1
        }
        resampled.withUnsafeBufferPointer {
            onAudio($0)
            inputTap?($0, adc)
        }
        return noErr
    }

    private func handleOutput(_ ts: UnsafePointer<AudioTimeStamp>, _ frames: UInt32,
                              _ data: UnsafeMutablePointer<AudioBufferList>) -> OSStatus {
        let list = UnsafeMutableAudioBufferListPointer(data)
        let out = UnsafeMutableBufferPointer(start: list[0].mData!.assumingMemoryBound(to: Float.self),
                                             count: Int(frames))
        let dac = hostSeconds(ts.pointee.mHostTime) + outLatency
        player.pull(out, dac: dac)
        outputTap?(UnsafeBufferPointer(out), dac)
        outputLevel = max(rms(UnsafeBufferPointer(out)), outputLevel * levelDecay(frames))
        return noErr
    }

    /// Peak-hold with a ~80 ms release, so readers sampling at screen rate see
    /// speech rather than whichever 2 ms block came last.
    private func levelDecay(_ frames: UInt32) -> Float {
        Float(exp(-Double(frames) / rate / 0.08))
    }

    private func rms(_ x: UnsafeBufferPointer<Float>) -> Float {
        guard !x.isEmpty else { return 0 }
        var sum: Float = 0
        for v in x { sum += v * v }
        return (sum / Float(x.count)).squareRoot()
    }

    /// A model frame from the server (`step` is the server's step time).
    func receive(index: Int, pcm: [Float], step: Double) {
        statsLock.lock()
        received[index] = now()
        serverStep[index] = step
        statsLock.unlock()
        player.push(index, pcm)
    }

    private func latencies() -> [Int] {
        os_unfair_lock_lock(&player.lock)
        let dac = player.dacStart, written = player.writtenAt
        os_unfair_lock_unlock(&player.lock)
        statsLock.lock()
        defer { statsLock.unlock() }
        return dac.keys.sorted().filter { j in
            j >= 1 && adcEnd[j - 1] != nil && received[j] != nil && written[j] != nil
        }
    }

    /// Median reply latency over the last frames played, in seconds. Copies only
    /// the player's short `recent` list, so the audio thread is not held up.
    func recentLatency() -> Double? {
        os_unfair_lock_lock(&player.lock)
        let recent = player.recent
        os_unfair_lock_unlock(&player.lock)
        statsLock.lock()
        let values = recent.compactMap { j, dac in j >= 1 ? adcEnd[j - 1].map { dac - $0 } : nil }
        statsLock.unlock()
        return values.isEmpty ? nil : median(values)
    }

    func report() -> LatencyReport {
        let frames = latencies()
        os_unfair_lock_lock(&player.lock)
        let dac = player.dacStart, written = player.writtenAt
        var r = LatencyReport()
        r.underruns = player.underruns
        r.caughtUpMs = Double(player.compressed) / rate * 1000
        os_unfair_lock_unlock(&player.lock)
        statsLock.lock()
        defer { statsLock.unlock() }
        let total = frames.map { dac[$0]! - adcEnd[$0 - 1]! }.sorted()
        guard !total.isEmpty else { return r }
        r.frames = total.count
        r.p50 = total[total.count / 2]
        r.p95 = total[min(total.count - 1, Int(Double(total.count) * 0.95))]
        r.minimum = total[0]
        r.maximum = total[total.count - 1]
        r.parts = [
            ("input (ADC -> callback)", median(frames.map { seen[$0 - 1]! - adcEnd[$0 - 1]! })),
            ("server (callback -> frame received)", median(frames.map { received[$0]! - seen[$0 - 1]! })),
            ("  of which step (server)", median(frames.map { serverStep[$0]! })),
            ("output queue (lead/margin)", median(frames.map { written[$0]! - received[$0]! })),
            ("output (render -> DAC)", median(frames.map { dac[$0]! - written[$0]! })),
        ]
        return r
    }
}
