import { FC, useCallback, useEffect } from "react";
import { useSocketContext } from "../../SocketContext";
import { useUserAudio } from "../../hooks/useUserAudio";

type UserAudioProps = {
  // The analyser on your voice, for the waveform lane.
  onAnalyser: (analyser: AnalyserNode | null) => void;
};
export const UserAudio: FC<UserAudioProps> = ({ onAnalyser }) => {
  const { sendMessage, isConnected } = useSocketContext();
  const onRecordingChunk = useCallback(
    (chunk: Uint8Array) => {
      if (!isConnected) {
        return;
      }
      sendMessage({
        type: "audio",
        data: chunk,
      });
    },
    [sendMessage, isConnected],
  );

  const { startRecordingUser, stopRecording } = useUserAudio({
    constraints: {
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
      video: false,
    },
    onDataChunk: onRecordingChunk,
  });

  useEffect(() => {
    let res: Awaited<ReturnType<typeof startRecordingUser>>;
    if (isConnected) {
      startRecordingUser().then(result => {
        if (result) {
          res = result;
          onAnalyser(result.analyser);
        }
      });
    }
    return () => {
      stopRecording();
      res?.source?.disconnect();
    };
  }, [startRecordingUser, stopRecording, isConnected, onAnalyser]);

  return null;
};
