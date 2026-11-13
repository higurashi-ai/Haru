// Command-line client for the model server
// (python -m haru_mlx.native). Uses the server's
// default card and the voice given with --voice; on exit it prints the reply
// latency and where it went.

import Foundation

struct Options {
    var socket = defaultSocket
    var input: String? = nil
    var output: String? = nil
    var bufferMs = 2.0
    var marginMs = 15.0
    var seconds: Double? = nil
    var voice: String? = nil
    var listDevices = false

    static func parse() -> Options {
        var o = Options()
        var args = CommandLine.arguments.dropFirst().makeIterator()
        while let a = args.next() {
            switch a {
            case "--socket": o.socket = args.next()!
            case "--input": o.input = args.next()!
            case "--output": o.output = args.next()!
            case "--buffer-ms": o.bufferMs = Double(args.next()!)!
            case "--margin-ms": o.marginMs = Double(args.next()!)!
            case "--seconds": o.seconds = Double(args.next()!)!
            case "--voice": o.voice = args.next()!
            case "--list-devices": o.listDevices = true
            default:
                print("usage: haru-client [--voice FILE] [--socket PATH] [--input NAME] [--output NAME] "
                    + "[--buffer-ms 2] [--margin-ms 15] [--seconds N] [--list-devices]")
                exit(2)
            }
        }
        return o
    }
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

let options = Options.parse()
if options.listDevices {
    for d in audioDevices() {
        print("\(d.name): \(d.inputs) in, \(d.outputs) out, \(Int(d.rate)) Hz")
    }
    exit(0)
}
guard let inDevice = findDevice(options.input, input: true) else { fail("no input device \(options.input ?? "")") }
guard let outDevice = findDevice(options.output, input: false) else { fail("no output device \(options.output ?? "")") }
guard let conn = Connection(path: options.socket) else {
    fail("cannot connect to \(options.socket): start the server first (python -m haru_mlx.native)")
}

guard let (assetsKind, _) = conn.readMessage(), assetsKind == Message.assets.rawValue else { fail("no assets") }
// The server's default card; the voice of --voice (a path the server reads).
conn.sendJSON(.start, options.voice.map { ["reference_path": URL(fileURLWithPath: $0).path] } ?? [:])
guard let (kind, payload) = conn.readMessage() else { fail("server closed") }
if kind == Message.error.rawValue { fail("server: \(decodeJSON(payload)["message"] ?? "error")") }
let hello = decodeJSON(payload)
let frameSamples = hello["frame_samples"] as! Int
let stepMs = hello["step_ms"] as! Double

let audio = DuplexAudio(input: inDevice, output: outDevice, bufferMs: options.bufferMs,
                        marginMs: options.marginMs, frameSamples: frameSamples, stepMs: stepMs,
                        onAudio: { conn.sendAudio($0) })

let done = DispatchSemaphore(value: 0)
Thread.detachNewThread {
    while let (kind, payload) = conn.readMessage() {
        if kind == Message.frame.rawValue {
            let f = decodeFrame(payload)
            audio.receive(index: f.index, pcm: f.pcm, step: f.stepSeconds)
        } else if kind == Message.text.rawValue {
            FileHandle.standardOutput.write(Data(payload))
        }
    }
    done.signal()
}

do { try audio.start() } catch { fail("\(error)") }
FileHandle.standardError.write("""
    input: \(inDevice.name) (\(Int(inDevice.rate)) Hz) | output: \(outDevice.name) (\(Int(outDevice.rate)) Hz) \
    | buffers in/out \(String(format: "%.1f/%.1f", audio.inputBufferMs, audio.outputBufferMs)) ms \
    | device latency in/out \(Int(audio.inLatency * 1000))/\(Int(audio.outLatency * 1000)) ms \
    | margin \(Int(audio.marginMs)) ms | lead \(Int(audio.lead * 1000)) ms (step \(Int(stepMs)) ms). \
    Talk now (Ctrl+C to stop).\n\n
    """.data(using: .utf8)!)
signal(SIGINT) { _ in done.signal() }
_ = done.wait(timeout: options.seconds.map { .now() + $0 } ?? .distantFuture)
let r = audio.report()  // before stopping: once the input stops, no more frames come
audio.stop()
conn.close()

if r.frames > 0 {
    print(String(format: "\n[latency] %d frames: user frame end (ADC) -> model reply (DAC) p50 %.0f ms, p95 %.0f ms, min %.0f ms, max %.0f ms | underruns %d, caught up %.0f ms",
                 r.frames, r.p50 * 1000, r.p95 * 1000, r.minimum * 1000, r.maximum * 1000,
                 r.underruns, r.caughtUpMs))
    print("[latency] p50 parts: " + r.parts.map { String(format: "%@ %.1f ms", $0.0, $0.1 * 1000) }
        .joined(separator: ", "))
}
