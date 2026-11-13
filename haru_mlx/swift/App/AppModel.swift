// State of the app: server, character card, voice reference, settings and the
// running conversation.

import AppKit
import AVFoundation
import SwiftUI

struct CardFields: Equatable {
    var persona = "", name = "", gender = "", backchannel = ""
    var otherName = "", relationship = "", situation = ""

    init() {}

    init(_ d: [String: Any]) {
        persona = d["persona"] as? String ?? ""
        name = d["name"] as? String ?? ""
        gender = d["gender"] as? String ?? ""
        backchannel = d["backchannel"] as? String ?? ""
        otherName = d["other_name"] as? String ?? ""
        relationship = d["relationship"] as? String ?? ""
        situation = d["situation"] as? String ?? ""
    }

    var json: [String: Any] {
        ["persona": persona, "name": name, "gender": gender, "backchannel": backchannel,
         "other_name": otherName, "relationship": relationship, "situation": situation]
    }
}

enum ReferenceSource: String, CaseIterable, Identifiable {
    case preset, file, recording
    var id: String { rawValue }

    var label: String {
        switch self {
        case .preset: return t("voice.default")
        case .file: return t("voice.file")
        case .recording: return t("voice.recording")
        }
    }
}

struct Preset: Identifiable {
    let id: String
    let serverLabel: String
    let card: [String: Any]

    /// The example's name in the UI language when the app has one, else the server's.
    var label: String { hasString("preset.\(id)") ? t("preset.\(id)") : serverLabel }
}

/// The latest user and model levels (0...1 on a -60..0 dBFS scale), read by
/// the meters at screen rate; not observable, so reading does not re-render
/// the views.
final class Levels {
    var source: () -> (Float, Float) = { (0, 0) }

    static func normalized(_ rms: Float) -> Float {
        let db = 20 * log10(max(rms, 1e-6))
        return max(0, min(1, (db + 60) / 60))
    }

    func current() -> (user: Float, model: Float) {
        let (u, m) = source()
        return (Self.normalized(u), Self.normalized(m))
    }
}

final class AppModel: ObservableObject {
    enum Phase { case launching, ready, starting, talking }

    // Server
    @Published var phase: Phase = .launching
    @Published var status = t("launch.connecting")
    @Published var serverLog: [String] = []
    @Published var errorMessage: String?
    @Published var precision = ""
    /// The model the server loaded (a repo id or a snapshot path), to reload it in another precision.
    private var modelPath = ""
    /// Restarting the server in another precision (its log keeps the reload status).
    private var reloading = false
    @Published var stepMs = 0.0

    // Character card
    @Published var languages: [(code: String, label: String)] = []
    /// The card's language, which the UI follows.
    @Published var language = UILanguage.current {
        didSet {
            UILanguage.current = language
            schedulePrepare()
        }
    }
    @Published var presets: [Preset] = []
    @Published var card = CardFields() { didSet { schedulePrepare() } }
    private var allPresets: [String: [Preset]] = [:]
    /// Default voice files the server has, by language (none as installed).
    private var voices: [String: String] = [:]

    // Voice reference
    @Published var referenceSource: ReferenceSource = .file { didSet { schedulePrepare() } }
    @Published var referenceFile: URL? { didSet { schedulePrepare() } }
    @Published var recordingURL: URL? { didSet { schedulePrepare() } }
    @Published var isRecording = false { didSet { schedulePrepare() } }
    @Published var recordingSeconds = 0.0
    private var recorder: AVAudioRecorder?
    private var preview: AVAudioPlayer?

    // Settings
    @Published var cfgScale = 1.5 { didSet { schedulePrepare() } }
    @Published var marginMs = 15.0
    @Published var inputDevices: [String] = []
    @Published var outputDevices: [String] = []
    @Published var inputDevice = ""
    @Published var outputDevice = ""

    // Prefill: the server prefills the current card and voice ahead of START.
    enum Prepare: Equatable { case idle, preparing, ready, failed(String) }
    @Published var prepare: Prepare = .idle
    private var prepareWork: DispatchWorkItem?
    private var preparedKey: String?  // the request of the last PREPARE
    private var prepareSeq = 0  // PREPAREs sent on this connection

    // Conversation
    @Published var transcript = ""
    @Published var latencyMs: Double?
    @Published var underruns = 0
    @Published var caughtUpMs = 0.0
    @Published var lastReport: LatencyReport?
    @Published var audioInfo = ""
    /// The last conversation as heard, for saving its audio and video.
    @Published var lastRecording: Recording?
    @Published var exportProgress: Double?

    private let launcher = ServerLauncher()
    private var conn: Connection?
    private var refusals = 0  // connections the server closed before sending its assets
    private var audio: DuplexAudio?
    private var capture: ConversationRecorder?
    /// Levels for the meters, read by the views at screen rate.
    let levels = Levels()
    /// Both voices and the model's text on the host clock, for the waveform lanes.
    let waves = WaveHistory()
    private var timer: Timer?
    /// Devices and margin for the conversation START asked for.
    private var plan: (input: Device, output: Device, margin: Double)?
    let socket = defaultSocket

    init() {
        NotificationCenter.default.addObserver(
            forName: NSApplication.willTerminateNotification, object: nil, queue: .main
        ) { [weak self] _ in
            self?.audio?.stop()
            self?.conn?.close()
            self?.launcher.stop()
        }
        refreshDevices()
    }

    // MARK: Server

    func launch() {
        AVCaptureDevice.requestAccess(for: .audio) { _ in }
        let info = Bundle.main.infoDictionary ?? [:]
        let python = info["HaruPython"] as? String ?? "/usr/bin/python3"
        launcher.connect(
            socket: socket, python: python, arguments: ["--language", language],
            onLog: logLine, onConnected: { [weak self] conn in self?.receiveAssets(conn) },
            onFailure: launchFailed)
    }

    /// Reload the current model in int8 or bf16: the server is restarted, the
    /// card, voice and settings stay.
    func setPrecision(_ precision: String) {
        guard precision != self.precision, phase == .ready, !isRecording, !modelPath.isEmpty else { return }
        let old = conn
        conn = nil
        old?.close()
        phase = .launching
        reloading = true
        status = t("launch.reloading", ["precision": precision == "bf16" ? "BF16" : "int8"])
        let info = Bundle.main.infoDictionary ?? [:]
        let python = info["HaruPython"] as? String ?? "/usr/bin/python3"
        launcher.restart(
            socket: socket, python: python,
            arguments: ["--model", modelPath, "--quantize", precision == "bf16" ? "none" : "8", "--language", language],
            onLog: logLine, onConnected: { [weak self] conn in self?.receiveAssets(conn) },
            onFailure: launchFailed)
    }

    private lazy var logLine: (String) -> Void = { [weak self] line in
        guard let self else { return }
        self.serverLog.append(line)
        fileLog(line)
        if self.serverLog.count > 300 { self.serverLog.removeFirst() }
        if self.phase == .launching, !self.reloading {
            self.status = t("launch.loading")
        }
    }

    private lazy var launchFailed: (String) -> Void = { [weak self] message in
        self?.status = t("launch.failed")
        self?.errorMessage = message
    }

    /// The server sends its card templates, presets and voices on connect,
    /// answers each PREPARE with READY, and START with HELLO (or ERROR), after
    /// which the conversation runs on this connection.
    private func receiveAssets(_ conn: Connection) {
        self.conn = conn
        prepareSeq = 0
        preparedKey = nil
        prepare = .idle
        DispatchQueue.global().async {
            guard let (kind, payload) = conn.readMessage(), kind == Message.assets.rawValue else {
                // The server turns clients away while the last one is still closing,
                // which a reconnect right after a conversation can hit: try again.
                DispatchQueue.main.async {
                    guard self.conn === conn else { return }
                    self.conn = nil
                    self.refusals += 1
                    if self.refusals <= 20 {
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { self.reconnect() }
                    } else {
                        self.errorMessage = t("error.busy")
                    }
                }
                return
            }
            DispatchQueue.main.async { self.refusals = 0 }
            let assets = decodeJSON(payload)
            DispatchQueue.main.async {
                self.apply(assets)
                self.schedulePrepare()
            }
            while let (kind, payload) = conn.readMessage() {
                switch kind {
                case Message.ready.rawValue:
                    let reply = decodeJSON(payload)
                    DispatchQueue.main.async { self.prepared(reply) }
                case Message.error.rawValue:
                    let message = decodeJSON(payload)["message"] as? String ?? t("error.title")
                    DispatchQueue.main.async { self.endConversation(error: message) }
                    return
                case Message.hello.rawValue:
                    self.converse(conn, hello: decodeJSON(payload))
                    return
                default:
                    break
                }
            }
            DispatchQueue.main.async {
                if self.conn === conn { self.endConversation(error: t("error.disconnected")) }
            }
        }
    }

    private func reconnect() {
        DispatchQueue.global().async {
            var conn: Connection?
            // Up to 60 s, so a server restarted with another model can finish loading.
            for _ in 0..<600 where conn == nil {
                conn = Connection(path: self.socket)
                if conn == nil { usleep(100_000) }
            }
            DispatchQueue.main.async {
                if let conn { self.receiveAssets(conn) } else {
                    self.errorMessage = t("error.reconnect")
                }
            }
        }
    }

    private func apply(_ assets: [String: Any]) {
        let firstTime = languages.isEmpty
        voices = assets["voices"] as? [String: String] ?? [:]
        precision = assets["precision"] as? String ?? ""
        modelPath = assets["model"] as? String ?? modelPath
        stepMs = assets["step_ms"] as? Double ?? 0
        languages = (assets["languages"] as? [[String: Any]] ?? []).map {
            ($0["code"] as? String ?? "", $0["label"] as? String ?? "")
        }
        let presetsJSON = assets["presets"] as? [String: [[String: Any]]] ?? [:]
        allPresets = presetsJSON.mapValues { list in
            list.map { Preset(id: $0["id"] as? String ?? "", serverLabel: $0["label"] as? String ?? "",
                              card: $0["card"] as? [String: Any] ?? [:]) }
        }
        if firstTime {
            language = assets["default_language"] as? String ?? language
            selectLanguage(language)
        }
        useDefaultVoiceIfAny()
        phase = .ready
        reloading = false
        status = ""
    }

    // MARK: Card

    func selectLanguage(_ code: String) {
        language = code
        useDefaultVoiceIfAny()
        presets = allPresets[code] ?? []
        if let first = presets.first { card = CardFields(first.card) }
    }

    func applyPreset(_ preset: Preset) { card = CardFields(preset.card) }

    // MARK: Voice reference

    var referenceURL: URL? {
        switch referenceSource {
        case .preset: return defaultVoice
        case .file: return referenceFile
        case .recording: return recordingURL
        }
    }

    /// The default voice of the card's language, if the server has one.
    var defaultVoice: URL? { voices[language].map { URL(fileURLWithPath: $0) } }

    /// The sources on offer: the default voice only when there is one.
    var referenceSources: [ReferenceSource] {
        ReferenceSource.allCases.filter { $0 != .preset || defaultVoice != nil }
    }

    /// Starts on the default voice when the language has one, and leaves it when not;
    /// a file or a recording the user chose stays.
    private func useDefaultVoiceIfAny() {
        if defaultVoice == nil {
            if referenceSource == .preset { referenceSource = .file }
        } else if referenceSource == .file, referenceFile == nil {
            referenceSource = .preset
        }
    }

    /// What is still missing before a conversation can start, as in the web client.
    var blocker: String? {
        if card.persona.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return t("state.personaMissing") }
        if referenceURL == nil { return t("state.voiceMissing") }
        return nil
    }

    func chooseFile() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio]
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK { referenceFile = panel.url }
    }

    func toggleRecording() {
        if isRecording {
            recorder?.stop()
            isRecording = false
            return
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("haru-reference-\(UUID().uuidString).wav")
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 48_000,
            AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16,
        ]
        do {
            recorder = try AVAudioRecorder(url: url, settings: settings)
            recorder?.record()
            recordingURL = url
            isRecording = true
            recordingSeconds = 0
            Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] t in
                guard let self, self.isRecording, let r = self.recorder else { t.invalidate(); return }
                self.recordingSeconds = r.currentTime
            }
        } catch {
            errorMessage = t("error.recording", ["error": error.localizedDescription])
        }
    }

    func playReference() {
        guard let url = referenceURL else { return }
        preview = try? AVAudioPlayer(contentsOf: url)
        preview?.play()
    }

    // MARK: Devices

    func refreshDevices() {
        let all = audioDevices()
        inputDevices = all.filter { $0.inputs > 0 }.map(\.name)
        outputDevices = all.filter { $0.outputs > 0 }.map(\.name)
        if inputDevice.isEmpty || !inputDevices.contains(inputDevice) {
            // A wired or USB microphone: the built-in one adds ~30 ms of processing,
            // and a Bluetooth one switches the headphones to call mode. Then the
            // built-in microphone.
            let skip = ["MacBook", "AirPods", "iPhone", "BlackHole", "Teams", "Zoom", "Bluetooth"]
            inputDevice = inputDevices.first { name in !skip.contains { name.contains($0) } }
                ?? inputDevices.first { $0.contains("MacBook") && $0.contains("Microphone") }
                ?? findDevice(nil, input: true)?.name ?? inputDevices.first ?? ""
        }
        if outputDevice.isEmpty || !outputDevices.contains(outputDevice) {
            outputDevice = findDevice(nil, input: false)?.name ?? outputDevices.first ?? ""
        }
    }

    // MARK: Saving

    func saveAudio() {
        guard let recording = lastRecording else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "haru audio.wav"
        panel.allowedContentTypes = [.wav]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do { try recording.writeAudio(to: url) } catch {
            errorMessage = t("error.saveAudio", ["error": error.localizedDescription])
        }
    }

    func saveVideo() {
        guard let recording = lastRecording, exportProgress == nil else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "haru video.mp4"
        panel.allowedContentTypes = [.mpeg4Movie]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        exportProgress = 0
        Task { @MainActor in
            do {
                try await recording.writeVideo(
                    to: url, userName: card.otherName.isEmpty ? t("speaker.you") : card.otherName,
                    modelName: card.name.isEmpty ? t("speaker.model") : card.name) { self.exportProgress = $0 }
            } catch {
                errorMessage = t("error.saveVideo", ["error": error.localizedDescription])
            }
            exportProgress = nil
        }
    }

    /// A warning when the chosen devices add a lot of latency.
    var deviceWarning: String? {
        var parts: [String] = []
        if outputDevice.contains("Speakers") { parts.append(t("devices.speakers")) }
        if inputDevice.contains("MacBook") { parts.append(t("devices.microphone")) }
        if (inputDevice + outputDevice).contains("AirPods") { parts.append(t("devices.bluetooth")) }
        return parts.isEmpty ? nil : t("devices.warning", ["parts": parts.joined(separator: t("devices.separator"))])
    }

    // MARK: Prefill

    /// START's request: the card, the language, the guidance scale and the voice.
    private var request: [String: Any] {
        var request: [String: Any] = ["card": card.json, "language": language, "guidance_scale": cfgScale]
        if let url = referenceURL { request["reference_path"] = url.path }
        return request
    }

    private func key(_ request: [String: Any]) -> String {
        let data = try? JSONSerialization.data(withJSONObject: request, options: .sortedKeys)
        return data.flatMap { String(data: $0, encoding: .utf8) } ?? ""
    }

    /// Has the server prefill the current settings 5 s after the last edit, so
    /// START answers at once (START prefills them itself if they changed since).
    func schedulePrepare() {
        prepareWork?.cancel()
        let work = DispatchWorkItem { [weak self] in self?.sendPrepare() }
        prepareWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 5, execute: work)
    }

    private func sendPrepare() {
        guard let conn, phase == .ready, !isRecording, blocker == nil else { return }
        let request = request, key = key(request)
        guard key != preparedKey else { return }
        preparedKey = key
        prepareSeq += 1
        prepare = .preparing
        conn.sendJSON(.prepare, request)
    }

    private func prepared(_ reply: [String: Any]) {
        guard reply["seq"] as? Int == prepareSeq, phase == .ready else { return }
        prepare = reply["ok"] as? Bool == true ? .ready : .failed(reply["message"] as? String ?? t("error.title"))
    }

    // MARK: Conversation

    /// Sends START; the screen switches once the server answers with the
    /// prefilled session (at once when PREPARE already prefilled it) and the
    /// audio is running.
    func startConversation() {
        guard let conn, phase == .ready, blocker == nil else { return }
        preview?.stop()
        guard let input = findDevice(inputDevice, input: true),
              let output = findDevice(outputDevice, input: false) else {
            errorMessage = t("error.noDevice")
            return
        }
        prepareWork?.cancel()
        plan = (input, output, marginMs)
        phase = .starting
        transcript = ""
        latencyMs = nil
        conn.sendJSON(.start, request)
    }

    /// Runs the conversation on the reader thread, from HELLO until the
    /// connection closes.
    private func converse(_ conn: Connection, hello: [String: Any]) {
        guard let (input, output, margin) = plan else { return }
        let audio = DuplexAudio(
            input: input, output: output, bufferMs: 2, marginMs: margin,
            frameSamples: hello["frame_samples"] as? Int ?? 6400,
            stepMs: hello["step_ms"] as? Double ?? 36, onAudio: { conn.sendAudio($0) })
        let recorder = ConversationRecorder()
        let waves = self.waves
        waves.reset()
        audio.inputTap = { x, adc in
            recorder.input(x, adc: adc)
            waves.input(x, at: adc)
        }
        audio.outputTap = { x, dac in
            recorder.output(x, dac: dac)
            waves.output(x, at: dac)
        }
        do {
            try audio.start()
        } catch {
            DispatchQueue.main.async { self.endConversation(error: t("error.audio", ["error": error])) }
            return
        }
        let info = String(format: "buffer input %.1f ms / output %.1f ms, margin %.0f ms",
                          audio.inputBufferMs, audio.outputBufferMs, audio.marginMs)
        fileLog("[app] start: input \(input.name) / output \(output.name) | " + info
                + String(format: " | lead %.0f ms", audio.lead * 1000))
        // Each device's own latency plus its buffer, next to its name.
        let devices = String(format: "%@ (%.1f ms) → %@ (%.1f ms)",
                             input.name, audio.inLatency * 1000 + audio.inputBufferMs,
                             output.name, audio.outLatency * 1000 + audio.outputBufferMs)
        DispatchQueue.main.async {
            self.audio = audio
            self.capture = recorder
            self.audioInfo = devices
            self.phase = .talking
            self.startTimer()
        }
        while let (kind, payload) = conn.readMessage() {
            if kind == Message.frame.rawValue {
                let f = decodeFrame(payload)
                audio.receive(index: f.index, pcm: f.pcm, step: f.stepSeconds)
            } else if kind == Message.text.rawValue, let text = String(bytes: payload, encoding: .utf8) {
                // Heard once its frame has waited out the margin and left the device.
                let heardAt = now() + audio.marginMs / 1000 + audio.outLatency
                recorder.text(text, heardAt: heardAt)
                waves.text(text, heardAt: heardAt)
                DispatchQueue.main.async { self.transcript += text }
            }
        }
        DispatchQueue.main.async { if self.phase == .talking { self.stopConversation() } }
    }

    func stopConversation() {
        guard phase == .talking, let audio else { return }
        let r = audio.report()  // before stopping: once the input stops, no more frames come
        lastReport = r
        fileLog(String(format: "[app] end: %d frames, latency p50 %.0f / p95 %.0f / max %.0f ms, underruns %d, caught up %.0f ms",
                       r.frames, r.p50 * 1000, r.p95 * 1000, r.maximum * 1000, r.underruns, r.caughtUpMs)
                + " | " + r.parts.map { String(format: "%@ %.1f", $0.0, $0.1 * 1000) }.joined(separator: ", "))
        audio.stop()
        self.audio = nil
        lastRecording = capture?.finish()
        capture = nil
        endConversation(error: nil)
    }

    private func endConversation(error: String?) {
        timer?.invalidate()
        timer = nil
        conn?.close()
        conn = nil
        if let error { errorMessage = error; fileLog("[app] error: \(error)") }
        phase = .launching
        status = t("launch.next")
        reconnect()
    }

    private func startTimer() {
        levels.source = { [weak self] in
            guard let audio = self?.audio else { return (0, 0) }
            return (audio.inputLevel, audio.outputLevel)
        }
        timer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
            guard let self, let audio = self.audio else { return }
            self.latencyMs = audio.recentLatency().map { $0 * 1000 }
            os_unfair_lock_lock(&audio.player.lock)
            self.underruns = audio.player.underruns
            self.caughtUpMs = Double(audio.player.compressed) / audio.rate * 1000
            os_unfair_lock_unlock(&audio.player.lock)
        }
    }
}

/// Appends a line to ~/Library/Logs/Haru/app.log (server output and conversation stats).
func fileLog(_ line: String) {
    logQueue.async {
        let dir = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Logs/Haru")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("app.log")
        let stamp = logFormatter.string(from: Date())
        let data = Data("\(stamp) \(line)\n".utf8)
        if let h = try? FileHandle(forWritingTo: url) {
            h.seekToEndOfFile()
            h.write(data)
            try? h.close()
        } else {
            try? data.write(to: url)
        }
    }
}

private let logQueue = DispatchQueue(label: "fileLog")
private let logFormatter: DateFormatter = {
    let f = DateFormatter()
    f.dateFormat = "yyyy-MM-dd HH:mm:ss"
    return f
}()
