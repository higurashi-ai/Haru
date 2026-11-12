import { useState, useEffect, useRef } from "react";
import type { AudioStats } from "../../hooks/useServerAudio";
import { useSocketContext } from "../../SocketContext";

type ServerAudioStatsProps = {
  getAudioStats: React.MutableRefObject<() => AudioStats>;
  sessionId?: string;
};

// The "Latency" readout (and what it is made of) goes to the server log too, every
// REPORT_MS and when the conversation ends. Only while connected: after a disconnect
// the player keeps running and would count its silence as missed audio, so the
// readout freezes at the last connected values.
const REPORT_MS = 10000;

// The last conversation's latency (seconds), shown on the setup page after it ends.
const LAST_LATENCY_KEY = "lastLatencyV1";
export type LastLatency = { p50: number; p95: number };

const storeLastLatency = (value: LastLatency) => {
  try {
    sessionStorage.setItem(LAST_LATENCY_KEY, JSON.stringify(value));
  } catch {
    // ignore
  }
};

export const loadLastLatency = (): LastLatency | null => {
  try {
    const value = JSON.parse(sessionStorage.getItem(LAST_LATENCY_KEY) ?? "null");
    return value && Number.isFinite(value.p50) && Number.isFinite(value.p95) ? value : null;
  } catch {
    return null;
  }
};

export const Stat = ({ label, value, big }: { label: string; value: string; big?: boolean }) => (
  <div className="flex flex-col gap-0.5">
    <span className="text-[11px] font-medium text-muted">{label}</span>
    <span className={`tabular-nums ${big ? "text-[26px] font-semibold leading-8 text-accent" : "text-base"}`}>{value}</span>
  </div>
);

export const ServerAudioStats = ({ getAudioStats, sessionId }: ServerAudioStatsProps) => {
  const [audioStats, setAudioStats] = useState(getAudioStats.current());
  const { isConnected } = useSocketContext();

  const movingAverageSum = useRef<number>(0.);
  const movingAverageCount = useRef<number>(0.);
  const movingBeta = 0.85;
  const samples = useRef<number[]>([]);  // displayed latency, one per tick once known

  let convertMinSecs = (total_secs: number) => {
    // convert secs to the format mm:ss.cc
    let mins = (Math.floor(total_secs / 60)).toString();
    let secs = (Math.floor(total_secs) % 60).toString();
    let cents = (Math.floor(100 * (total_secs - Math.floor(total_secs)))).toString();
    if (secs.length < 2) {
      secs = "0" + secs;
    }
    if (cents.length < 2) {
      cents = "0" + cents;
    }
    return mins + ":" + secs + "." + cents;
  };

  useEffect(() => {
    if (!isConnected) {
      return;
    }
    const interval = setInterval(() => {
      const newAudioStats = getAudioStats.current();
      setAudioStats(newAudioStats);
      movingAverageCount.current *= movingBeta;
      movingAverageCount.current += (1 - movingBeta) * 1;
      movingAverageSum.current *= movingBeta;
      movingAverageSum.current += (1 - movingBeta) * newAudioStats.delay;
      if (newAudioStats.delay > 0) {
        samples.current.push(movingAverageSum.current / movingAverageCount.current);
      }
    }, 141);
    const report = (final: boolean) => {
      const values = samples.current.slice().sort((a, b) => a - b);
      if (!sessionId || !values.length) return;
      if (final) {
        storeLastLatency({ p50: values[Math.floor(0.5 * values.length)], p95: values[Math.min(values.length - 1, Math.floor(0.95 * values.length))] });
      }
      const s = getAudioStats.current();
      const pick = (q: number) => values[Math.min(values.length - 1, Math.floor(q * values.length))];
      const body = JSON.stringify({
        session_id: sessionId,
        final,
        transport: s.transport ?? "websocket",
        latency_now: movingAverageSum.current / movingAverageCount.current,
        latency_p50: pick(0.5),
        latency_p95: pick(0.95),
        latency_min: values[0],
        latency_max: values[values.length - 1],
        rtt: s.rttSeconds ?? null,
        server_lag: s.serverLagSeconds ?? null,
        jitter_buffer: s.jitterBufferSeconds ?? null,
        capture: s.captureSeconds ?? null,
        playout_device: s.playoutDeviceSeconds ?? null,
        relay_protocol: s.relayProtocol ?? null,
        accelerated: s.acceleratedSeconds ?? null,
        decelerated: s.deceleratedSeconds ?? null,
        played: s.playedAudioDuration,
        missed: s.missedAudioDuration,
        down_gaps_250: s.downGaps250 ?? null,
        down_gap_max: s.downGapMaxSeconds ?? null,
      });
      try {
        navigator.sendBeacon("api/client_stats", new Blob([body], { type: "text/plain" }));
      } catch {
        // ignore
      }
    };
    const reporter = setInterval(() => report(false), REPORT_MS);
    return () => {
      clearInterval(interval);
      clearInterval(reporter);
      report(true);
    };
  }, [sessionId, isConnected]);

  const latency = movingAverageSum.current / movingAverageCount.current;
  const ms = (seconds: number) => `${Math.round(seconds * 1000)} ms`;
  return (
    <div className="flex flex-wrap items-end gap-x-7 gap-y-3 rounded-[18px] bg-card p-[18px]">
      <Stat big label="応答の遅延" value={Number.isFinite(latency) && latency > 0 ? ms(latency) : "—"} />
      <Stat label="再生済み" value={convertMinSecs(audioStats.playedAudioDuration)} />
      <Stat label="欠落" value={convertMinSecs(audioStats.missedAudioDuration)} />
      <Stat label="マイク → 出力デバイス" value={`${ms(audioStats.captureSeconds ?? 0)} → ${ms(audioStats.playoutDeviceSeconds ?? 0)}`} />
      <Stat label="バッファ（最小 / 最大）" value={audioStats.minPlaybackDelay <= audioStats.maxPlaybackDelay ? `${ms(audioStats.minPlaybackDelay)} / ${ms(audioStats.maxPlaybackDelay)}` : "—"} />
    </div>
  );
};
