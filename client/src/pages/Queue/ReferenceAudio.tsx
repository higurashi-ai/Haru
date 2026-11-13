import { FC, useCallback, useEffect, useRef, useState } from "react";
import { CircleCheck, LoaderCircle, Play, Square, TriangleAlert } from "lucide-react";
import { Button, SwitchButton } from "../../components/Button/Button";
import { getMimeType } from "../Conversation/getMimeType";
import { errorMessage, useT } from "../../i18n";

export type ReferenceAudioValue = {
  id: string;
  name: string;
  seconds: number;
  trimmed: boolean;
  // Only for this page load (a restored reference has no local preview).
  previewUrl?: string;
};

type ReferenceAudioProps = {
  value: ReferenceAudioValue | null;
  onChange: (value: ReferenceAudioValue | null) => void;
  maxSeconds: number;
  disabled?: boolean;
  defaultValue?: ReferenceAudioValue;
  // Language of the bundled voice, as shown to the viewer.
  defaultLabel?: string;
};

type Source = "preset" | "file" | "recording";
// Name given to a microphone recording (shown translated).
const RECORDING_NAME = "recording";

const formatSeconds = (seconds: number) => {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
};

// Voice reference of the model speaker: an uploaded file or a microphone
// recording, sent to /api/reference_audio (decoded, silence-trimmed and
// capped to the last `maxSeconds` on the server).
export const ReferenceAudio: FC<ReferenceAudioProps> = ({ value, onChange, maxSeconds, disabled, defaultValue, defaultLabel }) => {
  const t = useT();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const timer = useRef<number | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const upload = useCallback(async (blob: Blob, name: string) => {
    setIsUploading(true);
    setError(null);
    try {
      const response = await fetch("/api/reference_audio", {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: blob,
      });
      if (!response.ok) {
        throw new Error(errorMessage(t, await response.text(), "error.upload"));
      }
      const data = await response.json() as { id: string, seconds: number, trimmed: boolean };
      if (value?.previewUrl?.startsWith("blob:")) {
        URL.revokeObjectURL(value.previewUrl);
      }
      onChange({
        id: data.id,
        name,
        seconds: data.seconds,
        trimmed: data.trimmed,
        previewUrl: URL.createObjectURL(blob),
      });
    } catch (e) {
      console.error(e);
      setError(e instanceof Error && e.message ? e.message : t("error.upload"));
    } finally {
      setIsUploading(false);
    }
  }, [onChange, value, t]);

  const stopTimer = () => {
    if (timer.current != null) {
      window.clearInterval(timer.current);
      timer.current = null;
    }
  };

  const startRecording = useCallback(async () => {
    setError(null);
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      console.error(e);
      setError(t("error.microphone"));
      return;
    }
    const mimeType = getMimeType("audio");
    const mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
    chunks.current = [];
    mediaRecorder.ondataavailable = e => {
      if (e.data.size > 0) {
        chunks.current.push(e.data);
      }
    };
    mediaRecorder.onstop = () => {
      stream.getTracks().forEach(track => track.stop());
      const blob = new Blob(chunks.current, { type: mediaRecorder.mimeType || mimeType });
      chunks.current = [];
      upload(blob, RECORDING_NAME);
    };
    mediaRecorder.start();
    recorder.current = mediaRecorder;
    setIsRecording(true);
    setElapsed(0);
    const startedAt = Date.now();
    timer.current = window.setInterval(() => setElapsed((Date.now() - startedAt) / 1000), 200);
  }, [upload, t]);

  const stopRecording = useCallback(() => {
    stopTimer();
    recorder.current?.stop();
    recorder.current = null;
    setIsRecording(false);
  }, []);

  useEffect(() => () => {
    stopTimer();
    recorder.current?.stop();
  }, []);

  const clear = useCallback(() => {
    if (value?.previewUrl?.startsWith("blob:")) {
      URL.revokeObjectURL(value.previewUrl);
    }
    onChange(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }, [onChange, value]);

  // Which of the three sources is shown; the voice in use changes only when one is picked.
  const isDefault = !!value && value.id === defaultValue?.id;
  const [chosenSource, setSource] = useState<Source>(() => !value || isDefault ? "preset" : value.name === RECORDING_NAME ? "recording" : "file");
  // The default voice is offered only when the server has one for this language.
  const source: Source = chosenSource === "preset" && !defaultValue ? "file" : chosenSource;
  useEffect(() => {
    if (isDefault) {
      setSource("preset");
    }
  }, [isDefault]);

  // Preview of the voice in use.
  const player = useRef<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const stopPreview = useCallback(() => {
    player.current?.pause();
    player.current = null;
    setIsPlaying(false);
  }, []);
  useEffect(() => stopPreview, [value?.previewUrl, stopPreview]);
  const togglePreview = () => {
    if (isPlaying || !value?.previewUrl) {
      stopPreview();
      return;
    }
    const audio = new Audio(value.previewUrl);
    audio.onended = stopPreview;
    audio.onerror = stopPreview;
    player.current = audio;
    setIsPlaying(true);
    audio.play().catch(stopPreview);
  };

  const busy = disabled || isUploading;
  const ownName = value && !isDefault ? value.name : null;
  const sources: [Source, string][] = [["file", t("voice.file")], ["recording", t("voice.recording")]];
  if (defaultValue) {
    sources.unshift(["preset", t("voice.default")]);
  }
  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-1.5">
        {sources.map(([id, label]) => (
          <Button key={id} selected={source === id} disabled={busy || isRecording}
            onClick={() => {
              setSource(id);
              if (id === "preset" && defaultValue && !isDefault) {
                onChange(defaultValue);
              }
            }}>
            {label}
          </Button>
        ))}
        <Button className="ml-auto" disabled={!value?.previewUrl} onClick={togglePreview}>
          {isPlaying ? <Square className="h-3 w-3 fill-current" /> : <Play className="h-3 w-3 fill-current" />}
          {isPlaying ? t("voice.stop") : t("voice.preview")}
        </Button>
      </div>
      <input
        ref={fileInputRef}
        className="hidden"
        disabled={busy || isRecording}
        type="file"
        accept="audio/*"
        onChange={e => {
          const file = e.target.files?.[0];
          if (file) {
            upload(file, file.name);
          }
        }}
      />
      <div className="flex min-h-7 flex-wrap items-center gap-2.5 text-[13px] text-muted">
        {source === "preset" &&
          <span>{defaultLabel ? t("voice.defaultOf", { language: defaultLabel }) : t("voice.default")}</span>}
        {source === "file" && <>
          <span className="min-w-0 truncate">{ownName && ownName !== RECORDING_NAME ? ownName : t("voice.noFile")}</span>
          <Button className="ml-auto" disabled={busy} onClick={() => fileInputRef.current?.click()}>{t("voice.chooseFile")}</Button>
        </>}
        {source === "recording" && <>
          {isRecording && <span className="h-[9px] w-[9px] rounded-full bg-[#ff3b30]" />}
          <span className="tabular-nums">
            {isRecording ? formatSeconds(elapsed) : ownName === RECORDING_NAME && value ? t("voice.seconds", { seconds: value.seconds.toFixed(1) }) : t("voice.notRecorded")}
          </span>
          <Button className="ml-auto" selected={isRecording} disabled={busy && !isRecording}
            onClick={isRecording ? stopRecording : startRecording}>
            {isRecording ? t("voice.stopRecording") : t("voice.startRecording")}
          </Button>
        </>}
      </div>
      {isUploading && <p className="flex items-center gap-1.5 text-xs text-muted">
        <LoaderCircle className="h-3.5 w-3.5 animate-spin" />{t("voice.processing")}
      </p>}
      {error && <p className="flex items-start gap-1.5 text-xs text-warning">
        <TriangleAlert className="mt-px h-3.5 w-3.5 shrink-0" />{error}
      </p>}
      <div className="flex flex-wrap items-center gap-2.5 rounded-xl bg-paper px-3.5 py-2.5 text-[13px]">
        {value ? <>
          <CircleCheck className="h-3.5 w-3.5 shrink-0 text-accent" />
          <span className="min-w-0 flex-1 break-words">
            {t(value.trimmed ? "voice.inUseTrimmed" : "voice.inUse", {
              name: isDefault ? t("voice.default") : value.name === RECORDING_NAME ? t("voice.recording") : value.name,
              seconds: value.seconds.toFixed(1),
              max: maxSeconds,
            })}
          </span>
          {!disabled && !isDefault && <SwitchButton className="ml-auto" onClick={clear}>{t("voice.remove")}</SwitchButton>}
        </> : <span className="text-muted">{t("voice.none")}</span>}
      </div>
    </div>
  );
};
