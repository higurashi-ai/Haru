import { MutableRefObject, createContext, useContext } from "react";
type MediaContextType = {
  startRecording: () => void;
  stopRecording: () => void;
  audioContext: MutableRefObject<AudioContext>;
  audioStreamDestination: MutableRefObject<MediaStreamAudioDestinationNode>;
  // Feeds audioStreamDestination in stereo: connect(merger, 0, USER_CHANNEL / MODEL_CHANNEL).
  recordingMerger: MutableRefObject<ChannelMergerNode>;
  worklet: MutableRefObject<AudioWorkletNode>;
  micDuration: MutableRefObject<number>;
  // When micDuration last advanced (performance.now(), ms) and the microphone's input
  // latency (MediaTrackSettings.latency, s; 0 when the browser does not report it).
  micClock: MutableRefObject<{ pageAt: number; captureLatency: number }>;
  actualAudioPlayed: MutableRefObject<number>;
};

// Recordings put the user on the left channel and the model on the right.
export const USER_CHANNEL = 0;
export const MODEL_CHANNEL = 1;

export const MediaContext = createContext<MediaContextType | null>(null);

export const useMediaContext = () => {
  const context = useContext(MediaContext);
  if (!context) {
    throw new Error(
      "useMediaContext must be used within a MediaContextProvider",
    );
  }

  return context;
};