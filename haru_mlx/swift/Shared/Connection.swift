// Unix-socket protocol of the model server
// (python -m haru_mlx.native): a type byte, a
// little-endian u32 payload length, then the payload.

import Darwin
import Foundation

/// The model server's socket in this user's temporary directory ($TMPDIR), as
/// the Python server defaults to.
let defaultSocket = (NSTemporaryDirectory() as NSString).appendingPathComponent("haru.sock")

enum Message: UInt8 {
    case audio = 0x01, start = 0x02, prepare = 0x03
    case hello = 0x10, frame = 0x11, text = 0x12, assets = 0x13, error = 0x14, ready = 0x15
}

final class Connection {
    let fd: Int32
    private let writeLock = NSLock()

    init?(path: String) {
        fd = socket(AF_UNIX, SOCK_STREAM, 0)
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        withUnsafeMutableBytes(of: &addr.sun_path) { raw in
            _ = path.withCString {
                strncpy(raw.baseAddress!.assumingMemoryBound(to: CChar.self), $0, raw.count - 1)
            }
        }
        let status = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                connect(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        if status != 0 { return nil }  // deinit closes the fd
        var one: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, socklen_t(MemoryLayout<Int32>.size))
    }

    /// Wake any reader with EOF. The fd itself is closed in deinit, once no
    /// thread can still read it: closing it here would let the next connection
    /// reuse the number while a reader of this one is between reads.
    func close() {
        shutdown(fd, SHUT_RDWR)
    }

    deinit {
        Darwin.close(fd)
    }

    private func readExactly(_ count: Int) -> [UInt8]? {
        var data = [UInt8](repeating: 0, count: count)
        var got = 0
        while got < count {
            let n = data.withUnsafeMutableBytes { read(fd, $0.baseAddress! + got, count - got) }
            if n <= 0 { return nil }
            got += n
        }
        return data
    }

    /// The next message, or nil when the connection closed.
    func readMessage() -> (UInt8, [UInt8])? {
        guard let header = readExactly(5) else { return nil }
        let length = header[1...4].enumerated().reduce(0) { $0 | Int($1.element) << (8 * $1.offset) }
        guard let payload = readExactly(length) else { return nil }
        return (header[0], payload)
    }

    private func send(_ kind: Message, _ bytes: UnsafeRawBufferPointer) {
        var header = [UInt8](repeating: 0, count: 5)
        header[0] = kind.rawValue
        let length = UInt32(bytes.count)
        for i in 0..<4 { header[1 + i] = UInt8((length >> (8 * UInt32(i))) & 0xff) }
        writeLock.lock()
        defer { writeLock.unlock() }
        _ = header.withUnsafeBytes { write(fd, $0.baseAddress!, 5) }
        var offset = 0
        while offset < bytes.count {
            let n = write(fd, bytes.baseAddress! + offset, bytes.count - offset)
            if n <= 0 { return }
            offset += n
        }
    }

    func sendAudio(_ samples: UnsafeBufferPointer<Float>) {
        send(.audio, UnsafeRawBufferPointer(samples))
    }

    func sendJSON(_ kind: Message, _ object: [String: Any]) {
        let data = (try? JSONSerialization.data(withJSONObject: object)) ?? Data("{}".utf8)
        data.withUnsafeBytes { send(kind, $0) }
    }
}

func decodeJSON(_ payload: [UInt8]) -> [String: Any] {
    (try? JSONSerialization.jsonObject(with: Data(payload))) as? [String: Any] ?? [:]
}

/// FRAME payload: u32 index, f32 step ms, float32 PCM.
func decodeFrame(_ payload: [UInt8]) -> (index: Int, stepSeconds: Double, pcm: [Float]) {
    let index = payload[0..<4].enumerated().reduce(0) { $0 | Int($1.element) << (8 * $1.offset) }
    return payload.withUnsafeBytes { raw in
        let step = raw.load(fromByteOffset: 4, as: Float.self)
        let pcm = Array(UnsafeBufferPointer(
            start: (raw.baseAddress! + 8).assumingMemoryBound(to: Float.self),
            count: (payload.count - 8) / 4))
        return (index, Double(step) / 1000, pcm)
    }
}
