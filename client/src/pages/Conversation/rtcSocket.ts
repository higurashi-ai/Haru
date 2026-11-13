// WebRTC transport that looks like the /api/chat WebSocket to the rest of the client.
//
// Audio goes over the peer connection (microphone track up, model track down: Opus over
// UDP, so a late or lost packet is concealed instead of stalling the stream); model text
// and session events come over the "events" DataChannel and are re-emitted as the same
// binary protocol messages the WebSocket carries (0x00 handshake, 0x02 text, 0x06 ping),
// so useSocket / useServerText work unchanged. Audio "messages" sent by the Opus recorder
// are dropped: the microphone track already carries that audio.
//
// If WebRTC cannot connect (no UDP path, no TURN, server without WebRTC), it falls back
// to the /api/chat WebSocket transparently: the server is told to free the slot, and
// from then on messages and the recorder's Opus pages go through that WebSocket.

import { fetchTurnServers, loadTurnKey } from "./turnKey";

const CONNECT_TIMEOUT_MS = 10000;
// Pre-negotiated "events" DataChannel (both peers create it with this id; sent to the
// server as ``dcid``): no in-band open handshake, so events flow one round trip sooner.
const EVENTS_CHANNEL_ID = 4;

export type RtcAudioStats = {
  receivedSeconds: number;
  concealedSeconds: number;
  jitterBufferSeconds: number;
  packetsLost: number;
  // Same quantity the WebSocket path shows (mic time minus the stream time being played):
  // the server measures user stream time arrived minus model stream time sent; the
  // network round trip, this browser's jitter buffer and the audio devices (microphone
  // input latency, playout path after the jitter buffer) are added here. 0 until the
  // server has reported (the model has produced audio).
  latencySeconds: number;
  captureSeconds: number;
  playoutDeviceSeconds: number;
  rttSeconds: number;
  serverLagSeconds: number;
  relayProtocol: string;  // TURN leg: udp / tcp / tls ("" when not relayed)
  // NetEq time stretching (seconds of 48 kHz audio): removed by speeding up playout,
  // inserted by slowing it down; concealedSeconds above covers losses/late packets.
  acceleratedSeconds: number;
  deceleratedSeconds: number;
};

const encoder = new TextEncoder();
const textMessage = (text: string) => new Uint8Array([0x02, ...encoder.encode(text)]);

export class RtcSocket extends EventTarget {
  binaryType: BinaryType = "arraybuffer";
  readyState: number = WebSocket.CONNECTING;
  remoteStream: MediaStream | null = null;
  private pc: RTCPeerConnection | null = null;
  private channel: RTCDataChannel | null = null;
  private mic: MediaStream | null = null;
  private pingTimer: number | undefined;
  private connectTimer: number | undefined;
  private closedOnce = false;
  private ready = false;
  private fallbackWs: WebSocket | null = null;
  private serverLatency: { lag_ms: number } | null = null;
  private lastJitter = { delay: 0, emitted: 0, seconds: 0 };
  private lastPlayout = { delay: 0, samples: 0, seconds: -1 };
  transport: "webrtc" | "websocket" = "webrtc";

  constructor(
    private sessionId: string,
    private constraints: MediaStreamConstraints,
    private webSocketUrl: string,
  ) {
    super();
    // Same sequence as the WebSocket: "open" now, the handshake once the server is ready.
    setTimeout(() => {
      this.readyState = WebSocket.OPEN;
      this.dispatchEvent(new Event("open"));
      this.connect().catch(error => this.fallback(String(error)));
    }, 0);
  }

  private fallback(reason: string) {
    if (this.closedOnce || this.fallbackWs || this.ready) {
      if (!this.ready && !this.fallbackWs) this.fail(reason);
      return;
    }
    console.warn("[webrtc] falling back to WebSocket:", reason);
    this.transport = "websocket";
    window.clearTimeout(this.connectTimer);
    this.pc?.close();
    this.pc = null;
    this.mic?.getTracks().forEach(track => track.stop());
    // Free the slot the abandoned peer holds, then take the WebSocket path.
    fetch(`api/webrtc/cancel?session_id=${encodeURIComponent(this.sessionId)}`, { method: "POST" })
      .catch(() => undefined)
      .finally(() => {
        if (this.closedOnce) return;
        const ws = new WebSocket(this.webSocketUrl);
        ws.binaryType = "arraybuffer";
        ws.addEventListener("message", event => this.dispatchEvent(new MessageEvent("message", { data: event.data })));
        ws.addEventListener("close", () => this.close());
        this.fallbackWs = ws;
      });
  }

  private emit(data: Uint8Array) {
    this.dispatchEvent(new MessageEvent("message", { data: data.buffer }));
  }

  private fail(message: string) {
    console.error("[webrtc]", message);
    this.emit(textMessage(`[error] ${message}`));
    this.close();
  }

  private async connect() {
    const config = await (await fetch("api/webrtc/config", { cache: "no-store" })).json();
    // The visitor's own Cloudflare TURN key (UI) first, else the server's TURN, else WebSocket.
    const turnKey = config.client_turn ? loadTurnKey() : null;
    let iceServers: RTCIceServer[];
    let visitorIce: RTCIceServer[] | null = null;
    if (turnKey) {
      try {
        visitorIce = await fetchTurnServers(turnKey);
      } catch (error) {
        this.fallback(String(error));
        return;
      }
      iceServers = visitorIce;
    } else if (config.enabled) {
      iceServers = config.ice_servers ?? [];
    } else {
      this.fallback("WebRTC is not enabled on the server");
      return;
    }
    const pc = new RTCPeerConnection({ iceServers });
    this.pc = pc;
    this.mic = await navigator.mediaDevices.getUserMedia(this.constraints);
    this.mic.getAudioTracks().forEach(track => pc.addTrack(track, this.mic!));
    pc.ontrack = event => {
      // The server already paces and buffers the model audio: keep the browser's own playout
      // buffer at its minimum instead of stacking a second one on top.
      try {
        (event.receiver as any).jitterBufferTarget = 0;
      } catch {
        // older browsers
      }
      this.remoteStream = event.streams[0] ?? new MediaStream([event.track]);
      this.dispatchEvent(new Event("remotestream"));
    };
    pc.onconnectionstatechange = () => {
      if (pc.connectionState === "failed" && !this.ready) {
        this.fallback("ICE failed");
      } else if ((pc.connectionState === "failed" || pc.connectionState === "closed") && this.pc === pc) {
        this.close();
      }
    };
    const channel = pc.createDataChannel("events", { negotiated: true, id: EVENTS_CHANNEL_ID });
    this.channel = channel;
    channel.onmessage = event => {
      let message: any;
      try {
        message = JSON.parse(event.data);
      } catch {
        return;
      }
      if (message.type === "session.ready") {
        this.ready = true;
        window.clearTimeout(this.connectTimer);
        this.emit(new Uint8Array([0x00, 0x00, 0x00]));
        this.pingTimer = window.setInterval(() => this.emit(new Uint8Array([0x06])), 3000);
      } else if (message.type === "latency") {
        this.serverLatency = message;
      } else if (message.type === "text") {
        this.emit(textMessage(message.text));
      } else if (message.type === "error") {
        this.emit(textMessage(`[error] ${message.message}`));
      } else if (message.type === "session.done") {
        this.close();
      }
    };
    channel.onclose = () => {
      if (this.ready) this.close();
    };

    await pc.setLocalDescription(await pc.createOffer());
    // One-shot signalling: wait for ICE gathering (TURN relay candidates included).
    await new Promise<void>(resolve => {
      if (pc.iceGatheringState === "complete") return resolve();
      const done = () => pc.iceGatheringState === "complete" && resolve();
      pc.addEventListener("icegatheringstatechange", done);
      setTimeout(resolve, 4000);
    });
    const response = await fetch(`api/webrtc/offer?session_id=${encodeURIComponent(this.sessionId)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      // Only the generated short-lived credentials go to the server (its ICE agent needs them too).
      body: JSON.stringify({
        sdp: pc.localDescription!.sdp,
        type: pc.localDescription!.type,
        dcid: EVENTS_CHANNEL_ID,
        ...(visitorIce ? { ice_servers: visitorIce } : {}),
      }),
    });
    const answer = await response.json();
    if (!response.ok) {
      throw new Error(answer.error ?? `HTTP ${response.status}`);
    }
    await pc.setRemoteDescription(answer);
    // The server holds a slot from here on: ICE + DTLS + "session.ready" must follow quickly.
    this.connectTimer = window.setTimeout(() => {
      if (!this.ready) this.fallback("no WebRTC connection within 10 s");
    }, CONNECT_TIMEOUT_MS);
  }

  // The Opus recorder keeps running for the visualizer/recording; over WebRTC its pages are
  // not needed, after a fallback they are the user's audio.
  send(data: ArrayBufferLike | ArrayBufferView | Blob | string) {
    if (this.fallbackWs?.readyState === WebSocket.OPEN) {
      this.fallbackWs.send(data);
    }
  }

  // fallbackPlayoutSeconds: the output device latency to use when the browser has no
  // "media-playout" stats (AudioContext.outputLatency).
  async getAudioStats(fallbackPlayoutSeconds = 0): Promise<RtcAudioStats | null> {
    if (!this.pc || this.fallbackWs) return null;
    const report = await this.pc.getStats();
    let stats: RtcAudioStats | null = null;
    let rtt = 0;
    let localCandidate = "";
    report.forEach((s: any) => {
      if (s.type === "candidate-pair" && s.nominated && s.state === "succeeded" && s.currentRoundTripTime) {
        rtt = s.currentRoundTripTime;
        localCandidate = s.localCandidateId;
      }
    });
    const local: any = localCandidate ? report.get(localCandidate) : null;
    const relayProtocol = local?.candidateType === "relay" ? String(local.relayProtocol ?? "") : "";
    report.forEach((s: any) => {
      // Chrome: delay from leaving the jitter buffer to the audio device, over the last interval.
      if (s.type === "media-playout" && s.kind === "audio") {
        const delay = s.totalPlayoutDelay ?? 0;
        const samples = s.totalSamplesCount ?? 0;
        if (samples > this.lastPlayout.samples) {
          this.lastPlayout = { delay, samples, seconds: (delay - this.lastPlayout.delay) / (samples - this.lastPlayout.samples) };
        }
      }
    });
    const settings: any = this.mic?.getAudioTracks()[0]?.getSettings() ?? {};
    const capture = Number(settings.latency) || 0;
    const playout = this.lastPlayout.seconds >= 0 ? this.lastPlayout.seconds : fallbackPlayoutSeconds;
    report.forEach((s: any) => {
      if (s.type === "inbound-rtp" && s.kind === "audio") {
        const rate = 48000;
        // Jitter buffer over the last interval (the cumulative mean hides changes).
        const delay = s.jitterBufferDelay ?? 0;
        const emitted = s.jitterBufferEmittedCount ?? 0;
        if (emitted > this.lastJitter.emitted) {
          this.lastJitter = { delay, emitted, seconds: (delay - this.lastJitter.delay) / (emitted - this.lastJitter.emitted) };
        }
        const b = this.serverLatency;
        stats = {
          receivedSeconds: (s.totalSamplesReceived ?? 0) / rate,
          concealedSeconds: (s.concealedSamples ?? 0) / rate,
          jitterBufferSeconds: this.lastJitter.seconds,
          packetsLost: s.packetsLost ?? 0,
          latencySeconds: b ? rtt + b.lag_ms / 1000 + this.lastJitter.seconds + capture + playout : 0,
          captureSeconds: capture,
          playoutDeviceSeconds: playout,
          rttSeconds: rtt,
          acceleratedSeconds: (s.removedSamplesForAcceleration ?? 0) / rate,
          deceleratedSeconds: (s.insertedSamplesForDeceleration ?? 0) / rate,
          serverLagSeconds: b ? b.lag_ms / 1000 : 0,
          relayProtocol,
        };
      }
    });
    return stats;
  }

  close() {
    if (this.closedOnce) return;
    this.closedOnce = true;
    window.clearInterval(this.pingTimer);
    window.clearTimeout(this.connectTimer);
    this.fallbackWs?.close();
    try {
      if (this.channel?.readyState === "open") {
        this.channel.send(JSON.stringify({ type: "session.end" }));
      }
    } catch {
      // ignore
    }
    setTimeout(() => this.pc?.close(), 300);
    this.mic?.getTracks().forEach(track => track.stop());
    this.readyState = WebSocket.CLOSED;
    this.dispatchEvent(new Event("close"));
  }
}
