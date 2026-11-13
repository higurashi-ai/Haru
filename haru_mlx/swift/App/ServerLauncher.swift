// Starts the Python model server when none is listening, and stops it with the app.

import Foundation

final class ServerLauncher {
    private var process: Process?
    private var ready = false

    /// Connect to a running server, or start one and connect once it is ready.
    /// Callbacks run on the main queue.
    func connect(socket: String, python: String, arguments: [String],
                 onLog: @escaping (String) -> Void,
                 onConnected: @escaping (Connection) -> Void,
                 onFailure: @escaping (String) -> Void) {
        if let conn = Connection(path: socket) {
            onConnected(conn)
            return
        }
        start(socket: socket, python: python, arguments: arguments,
              onLog: onLog, onConnected: onConnected, onFailure: onFailure)
    }

    /// Stop whichever server is running (this app's or one started elsewhere)
    /// and start one with these arguments.
    func restart(socket: String, python: String, arguments: [String],
                 onLog: @escaping (String) -> Void,
                 onConnected: @escaping (Connection) -> Void,
                 onFailure: @escaping (String) -> Void) {
        stop()
        DispatchQueue.global().async {
            let pattern = "haru_mlx.native"
            Self.run("/usr/bin/pkill", ["-f", pattern])
            for _ in 0..<100 where Self.run("/usr/bin/pgrep", ["-f", pattern]) == 0 { usleep(100_000) }
            DispatchQueue.main.async {
                self.start(socket: socket, python: python, arguments: arguments,
                           onLog: onLog, onConnected: onConnected, onFailure: onFailure)
            }
        }
    }

    @discardableResult private static func run(_ tool: String, _ args: [String]) -> Int32 {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: tool)
        p.arguments = args
        p.standardOutput = FileHandle.nullDevice
        p.standardError = FileHandle.nullDevice
        guard (try? p.run()) != nil else { return -1 }
        p.waitUntilExit()
        return p.terminationStatus
    }

    private func start(socket: String, python: String, arguments: [String],
                       onLog: @escaping (String) -> Void,
                       onConnected: @escaping (Connection) -> Void,
                       onFailure: @escaping (String) -> Void) {
        ready = false
        let p = Process()
        p.executableURL = URL(fileURLWithPath: python)
        p.arguments = ["-m", "haru_mlx.native", "--socket", socket] + arguments
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        p.environment = env
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        var pending = ""
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            pending += text
            while let newline = pending.firstIndex(where: { $0 == "\n" || $0 == "\r" }) {
                let line = String(pending[..<newline]).trimmingCharacters(in: .whitespaces)
                pending = String(pending[pending.index(after: newline)...])
                guard !line.isEmpty else { continue }
                DispatchQueue.main.async { onLog(line) }
                if line.contains("[native] ready on"), let self, !self.ready {
                    self.ready = true
                    DispatchQueue.global().async {
                        let conn = Connection(path: socket)
                        DispatchQueue.main.async {
                            if let conn { onConnected(conn) } else { onFailure(t("error.connect")) }
                        }
                    }
                }
            }
        }
        p.terminationHandler = { [weak self] proc in
            guard let self, !self.ready, self.process === proc else { return }
            DispatchQueue.main.async {
                onFailure(t("error.exited", ["code": proc.terminationStatus]))
            }
        }
        do {
            try p.run()
            process = p
            onLog("$ \(python) -m haru_mlx.native \(arguments.joined(separator: " "))")
        } catch {
            onFailure(t("error.launch", ["error": error.localizedDescription]) + "\nPython: \(python)")
        }
    }

    /// Stop the server if this app started it.
    func stop() {
        if let p = process, p.isRunning { p.terminate() }
        process = nil
    }
}
