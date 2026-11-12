import { useCallback, useEffect, useRef, useState } from "react";
import { useSocketContext } from "../SocketContext";
import { decodeMessage } from "../../../protocol/encoder";
import { MODEL_CHANNEL, useMediaContext } from "../MediaContext";
import { DecoderWorker } from "../../../decoder/decoderWorker";
import { RtcSocket } from "../rtcSocket";
import { getPlayoutProfile } from "../playoutProfile";

export type AudioStats = {
  playedAudioDuration: number;
  missedAudioDuration: number;
  totalAudioMessages: number;
  delay: number;
  minPlaybackDelay: number;
  maxPlaybackDelay: number;
  transport?: "webrtc" | "websocket";
  rttSeconds?: number;
  serverLagSeconds?: number;
  jitterBufferSeconds?: number;
  // Audio devices, included in delay: microphone input latency and the output path
  // (WebRTC: after the jitter buffer; WebSocket: AudioContext base + output latency).
  captureSeconds?: number;
  playoutDeviceSeconds?: number;
  relayProtocol?: string;
  acceleratedSeconds?: number;
  deceleratedSeconds?: number;
  // WebSocket: gaps between arriving model audio messages (downlink + server send timing).
  downGaps250?: number;
  downGapMaxSeconds?: number;
};

type useServerAudioArgs = {
  setGetAudioStats?: (getAudioStats: () => AudioStats) => void;
};

type WorkletStats = {
  totalAudioPlayed: number;
  actualAudioPlayed: number;
  timeInStream?: number;  // played stream time (drops and catch-up count as played)
  playing?: boolean;
  at?: number;  // AudioContext time of the post
  minDelay: number;
  maxDelay: number;
  compressedAudio?: number;  // catch-up: seconds removed by the worklet's WSOLA
};

export const useServerAudio = ({setGetAudioStats}: useServerAudioArgs) => {
  const { socket  } = useSocketContext();
  const {startRecording, stopRecording, audioContext, worklet, micDuration, micClock, actualAudioPlayed, recordingMerger } =
    useMediaContext();
  const analyser = useRef(audioContext.current.createAnalyser());
  worklet.current.connect(analyser.current);
  const decoderWorker = useRef(DecoderWorker);
  const [hasCriticalDelay, setHasCriticalDelay] = useState(false);
  const totalAudioMessages = useRef(0);
  const receivedDuration = useRef(0);
  const lastAudioArrival = useRef<number | null>(null);
  const downGaps250 = useRef(0);
  const downGapMax = useRef(0);
  const workletStats = useRef<WorkletStats>({
    totalAudioPlayed: 0,
    actualAudioPlayed: 0,
    minDelay: 0,
    maxDelay: 0,});

  const onDecode = useCallback(
    async (data: Float32Array) => {
      receivedDuration.current += data.length / audioContext.current.sampleRate;
      worklet.current.port.postMessage({frame: data, type: "audio"});
    },
    [],
  );

  const onWorkletMessage = useCallback(
    (event: MessageEvent<WorkletStats>) => {
      workletStats.current = event.data;
      actualAudioPlayed.current = workletStats.current.actualAudioPlayed;
    },
    [],
  );
  worklet.current.port.onmessage = onWorkletMessage;

  // WebRTC: the browser's own jitter buffer plays the model track; its stats stand in
  // for the worklet's (concealed samples are the "missed" audio).
  const rtcStats = useRef<AudioStats | null>(null);

  const outputLatency = () => {
    const ctx: any = audioContext.current;
    return (Number(ctx.baseLatency) || 0) + (Number(ctx.outputLatency) || 0);
  };

  // WebSocket latency: mic time minus played stream time, both extrapolated to now (the
  // encoder advances micDuration once per Ogg page, the worklet posts every ~20 ms), plus
  // the audio devices. 0 until model audio has played.
  const webSocketDelay = () => {
    const w = workletStats.current;
    if (!w.timeInStream) return 0;
    const sincePage = micClock.current.pageAt ? (performance.now() - micClock.current.pageAt) / 1000 : 0;
    const micNow = micDuration.current + Math.min(0.06, Math.max(0, sincePage));
    const sincePost = w.playing && w.at !== undefined ? audioContext.current.currentTime - w.at : 0;
    const playedNow = w.timeInStream + Math.min(0.03, Math.max(0, sincePost));
    return micNow - playedNow + micClock.current.captureLatency + outputLatency();
  };

  const getAudioStats = useCallback(() => {
    if (rtcStats.current) {
      return rtcStats.current;
    }
    return {
      playedAudioDuration: workletStats.current.actualAudioPlayed,
      delay: webSocketDelay(),
      captureSeconds: micClock.current.captureLatency,
      playoutDeviceSeconds: outputLatency(),
      minPlaybackDelay: workletStats.current.minDelay,
      maxPlaybackDelay: workletStats.current.maxDelay,
      missedAudioDuration: workletStats.current.totalAudioPlayed - workletStats.current.actualAudioPlayed,
      totalAudioMessages: totalAudioMessages.current,
      transport: "websocket" as const,
      acceleratedSeconds: workletStats.current.compressedAudio ?? 0,
      downGaps250: downGaps250.current,
      downGapMaxSeconds: downGapMax.current / 1000,
    };
  }, []);

  const onWorkerMessage = useCallback(
    (e: MessageEvent<any>) => {
      if (!e.data) {
        return;
      }
      onDecode(e.data[0]);
    },
    [onDecode],
  );

  const decodeAudio = useCallback((data: Uint8Array) => {
    decoderWorker.current.postMessage(
      {
        command: "decode",
        pages: data,
      },
      [data.buffer],
    );
  }, []);

  const onSocketMessage = useCallback(
    (e: MessageEvent) => {
      const dataArray = new Uint8Array(e.data);
      const message = decodeMessage(dataArray);
      // The recording starts when the conversation does (handshake, or the first
      // model audio), not at connect: that would capture the setup as silence.
      if (message.type === "handshake" || message.type === "audio") {
        startRecording();
      }
      if (message.type === "audio") {
        const now = performance.now();
        if (lastAudioArrival.current !== null) {
          const gap = now - lastAudioArrival.current;
          if (gap > 250) downGaps250.current++;
          downGapMax.current = Math.max(downGapMax.current, gap);
        }
        lastAudioArrival.current = now;
        decodeAudio(message.data);
        totalAudioMessages.current++;
      }
    },
    [decodeAudio, startRecording],
  );

  useEffect(() => {
    // Reset when the next connection starts, not in the cleanup: the stats panel's final
    // report at disconnect still reads the WebRTC stats.
    rtcStats.current = null;
    const currentSocket = socket as unknown;
    if (!(currentSocket instanceof RtcSocket)) {
      return;
    }
    const rtc = currentSocket;
    let element: HTMLAudioElement | null = null;
    let source: MediaStreamAudioSourceNode | null = null;
    const onRemoteStream = () => {
      if (!rtc.remoteStream) return;
      // Played by an <audio> element (needed for Chrome to pull WebRTC audio, and the
      // echo canceller sees it); WebAudio taps the same stream for visualizer + recording.
      element = new Audio();
      element.autoplay = true;
      element.srcObject = rtc.remoteStream;
      element.play().catch(error => console.error("[webrtc] playback", error));
      source = audioContext.current.createMediaStreamSource(rtc.remoteStream);
      source.connect(analyser.current);
      source.connect(recordingMerger.current, 0, MODEL_CHANNEL);
    };
    rtc.addEventListener("remotestream", onRemoteStream);
    const timer = window.setInterval(async () => {
      const stats = await rtc.getAudioStats(Number((audioContext.current as any).outputLatency) || 0);
      if (!stats) {
        rtcStats.current = null;  // fell back to the WebSocket: worklet stats again
        return;
      }
      actualAudioPlayed.current = stats.receivedSeconds - stats.concealedSeconds;
      const known = stats.latencySeconds > 0;
      rtcStats.current = {
        playedAudioDuration: stats.receivedSeconds - stats.concealedSeconds,
        missedAudioDuration: stats.concealedSeconds,
        totalAudioMessages: 0,
        delay: stats.latencySeconds,
        minPlaybackDelay: known ? Math.min(rtcStats.current?.minPlaybackDelay || Infinity, stats.latencySeconds) : 0,
        maxPlaybackDelay: known ? Math.max(rtcStats.current?.maxPlaybackDelay ?? 0, stats.latencySeconds) : 0,
        transport: "webrtc",
        rttSeconds: stats.rttSeconds,
        serverLagSeconds: stats.serverLagSeconds,
        jitterBufferSeconds: stats.jitterBufferSeconds,
        captureSeconds: stats.captureSeconds,
        playoutDeviceSeconds: stats.playoutDeviceSeconds,
        relayProtocol: stats.relayProtocol,
        acceleratedSeconds: stats.acceleratedSeconds,
        deceleratedSeconds: stats.deceleratedSeconds,
      };
    }, 500);
    return () => {
      rtc.removeEventListener("remotestream", onRemoteStream);
      window.clearInterval(timer);
      source?.disconnect();
      if (element) {
        element.pause();
        element.srcObject = null;
      }
    };
  }, [socket]);

  useEffect(() => {
    const currentSocket = socket;
    if (!currentSocket) {
      return;
    }
    const playout = getPlayoutProfile();
    worklet.current.port.postMessage({
      type: "reset",
      startMs: playout.playout_start_ms,
      catchupTargetMs: playout.catchup_target_ms,
    });
    currentSocket.addEventListener("message", onSocketMessage);
    totalAudioMessages.current = 0;
    return () => {
      stopRecording();
      currentSocket.removeEventListener("message", onSocketMessage);
    };
  }, [socket]);

  useEffect(() => {
    if (setGetAudioStats) {
      setGetAudioStats(getAudioStats);
    }
  }, [setGetAudioStats, getAudioStats]);

  useEffect(() => {
    decoderWorker.current.onmessage = onWorkerMessage;
    // 960 = 24000 / 12.5 / 2
    // The /2 is a bit optional, but won't hurt for recording the mic, and for the
    // the decoding it might help getting some decoded audio out asap.
    decoderWorker.current.postMessage({
      command: "init",
      bufferLength: 960 * audioContext.current.sampleRate / 24000,
      decoderSampleRate: 24000,
      outputBufferSampleRate: audioContext.current.sampleRate,
      resampleQuality: 0,
    });

  }, [onWorkerMessage]);

  return {
    decodeAudio,
    analyser,
    getAudioStats,
    hasCriticalDelay,
    setHasCriticalDelay,
  };
};
