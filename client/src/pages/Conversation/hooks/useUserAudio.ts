import { useCallback, useRef, useState } from "react";
import { getPlayoutProfile } from "../playoutProfile";
import Recorder from "opus-recorder";
import encoderPath from "opus-recorder/dist/encoderWorker.min.js?url";
import { USER_CHANNEL, useMediaContext } from "../MediaContext";

export enum UserMediaStatuses {
  IDLE = "IDLE",
  READY = "READY",
  WAITING_FOR_PERMISSION = "WAITING_FOR_PERMISSION",
  ERROR = "ERROR",
  RECORDING = "RECORDING",
  STOPPED = "STOPPED",
  STOPPING = "STOPPING",
}

type useUserAudioArgs = {
  constraints: MediaStreamConstraints;
  onDataChunk?: (chunk: Uint8Array) => void;
  onRecordingStart?: () => void;
  onRecordingStop?: () => void;
};

// One shared default: a new function per render would restart the recorder on every render.
const noop = () => {};

export const useUserAudio = ({
  constraints,
  onDataChunk,
  onRecordingStart = noop,
  onRecordingStop = noop,
}: useUserAudioArgs) => {
  const { recordingMerger, audioContext, micDuration, micClock } = useMediaContext();
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<UserMediaStatuses>(
    UserMediaStatuses.IDLE,
  );

  const recorder = useRef<any>(null);

  const getMediaStream = useCallback(async () => {
    setStatus(UserMediaStatuses.WAITING_FOR_PERMISSION);
    try {
      const stream =
        await window.navigator.mediaDevices.getUserMedia(constraints);
      setStatus(UserMediaStatuses.IDLE);
      return stream;
    } catch (error: any) {
      console.error(error);
      setError(error.name);
      setStatus(UserMediaStatuses.ERROR);
      return null;
    }
  }, [constraints, setStatus]);

  const startRecordingUser = useCallback(async () => {
    const mediaStream = await getMediaStream();
    if (mediaStream) {
      const settings: any = mediaStream.getAudioTracks()[0]?.getSettings() ?? {};
      micClock.current = { pageAt: 0, captureLatency: Number(settings.latency) || 0 };
      const analyser = audioContext.current.createAnalyser();
      const source = audioContext.current.createMediaStreamSource(mediaStream);
      source.connect(analyser);
      source.connect(recordingMerger.current, 0, USER_CHANNEL);

      // For buffer length: 960 = 24000 / 12.5 / 2
      // The /2 is a bit optional, but won't hurt for recording the mic.
      // Note that bufferLength actually has 0 impact for mono audio, only
      // the frameSize and maxFramesPerPage seems to have any.
      const recorderOptions = {
        mediaTrackConstraints: constraints,
        encoderPath,
        bufferLength: Math.round(960 * audioContext.current.sampleRate / 24000),
        encoderFrameSize: 20,
        encoderSampleRate: 24000,
        maxFramesPerPage: getPlayoutProfile().mic_frames_per_page,
        numberOfChannels: 1,
        recordingGain: 1,
        resampleQuality: 3,
        encoderComplexity: 0,
        encoderApplication: 2049,
        streamPages: true,
      };
      recorder.current = new Recorder(recorderOptions);
      recorder.current.ondataavailable = (data: Uint8Array) => {
        // Opus positions are always counted at 48 kHz.
        micDuration.current = recorder.current.encodedSamplePosition / 48000;
        micClock.current.pageAt = performance.now();
        if (onDataChunk) {
          onDataChunk(data);
        }
      };
      recorder.current.onstart = () => {
        setStatus(UserMediaStatuses.RECORDING);
        onRecordingStart();
      };
      recorder.current.onstop = () => {
        setStatus(UserMediaStatuses.STOPPED);
        source.disconnect();
        onRecordingStop();

        recorder.current = null;
      };

      if (recorder.current) {
        recorder.current.start();

      }

      return {
        analyser,
        mediaStream,
        source,
      };
    }
    return {
      analyser: null,
      mediaStream: null,
      source: null,
    };
  }, [setStatus, onDataChunk, onRecordingStart, onRecordingStop]);

  const stopRecording = useCallback(() => {
    setStatus(UserMediaStatuses.STOPPING);
    if (recorder.current) {
      recorder.current.stop();
    }
  }, [setStatus]);

  return {
    status,
    error,
    startRecordingUser,
    stopRecording,
  };
};
