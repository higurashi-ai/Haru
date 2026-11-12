import { FC, MutableRefObject, useCallback, useEffect, useRef, useState } from "react";
import { useSocket } from "./hooks/useSocket";
import { SocketContext } from "./SocketContext";
import { ServerAudio } from "./components/ServerAudio/ServerAudio";
import { UserAudio } from "./components/UserAudio/UserAudio";
import { AudioLines, Film } from "lucide-react";
import { Button, capsuleClass } from "../../components/Button/Button";
import { Masthead } from "../../components/Haru/Masthead";
import { WaveLanes } from "./components/WaveLanes/WaveLanes";
import { ServerAudioStats } from "./components/ServerAudio/ServerAudioStats";
import { AudioStats } from "./hooks/useServerAudio";
import { TextDisplay } from "./components/TextDisplay/TextDisplay";
import { MODEL_CHANNEL, MediaContext } from "./MediaContext";
import { ModelParamsValues, useModelParams } from "./hooks/useModelParams";
import fixWebmDuration from "webm-duration-fix";
import { getMimeType, getExtension } from "./getMimeType";
import { RtcSocket } from "./rtcSocket";
import { VIDEO_PIXELS } from "../../haru/painters";

const MIC_CONSTRAINTS: MediaStreamConstraints = {
  audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
  video: false,
};

type ConversationProps = {
  audioContext: MutableRefObject<AudioContext>;
  worklet: MutableRefObject<AudioWorkletNode>;
  // Audio over WebRTC (Opus/UDP) instead of the /api/chat WebSocket.
  useWebRtc?: boolean;
  // Names on the two lanes (from the character card).
  userName?: string;
  modelName?: string;
} & Partial<ModelParamsValues>;

// The prompt, voice and settings were registered with POST /api/session; the socket
// only carries the session id.
const chatURL = (sessionId: string) => {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const url = new URL(`${protocol}://${window.location.host}/api/chat`);
  url.searchParams.append("session_id", sessionId);
  return url.toString();
};

// Pixel size of the exported video.
const VIDEO_SIZE = VIDEO_PIXELS;

export const Conversation: FC<ConversationProps> = ({
  audioContext,
  worklet,
  useWebRtc = false,
  userName,
  modelName,
  ...params
}) => {
  const getAudioStats = useRef<() => AudioStats>(() => ({
    playedAudioDuration: 0,
    missedAudioDuration: 0,
    totalAudioMessages: 0,
    delay: 0,
    minPlaybackDelay: 0,
    maxPlaybackDelay: 0,
  }));
  const isRecording = useRef<boolean>(false);
  const videoChunks = useRef<Blob[]>([]);
  const audioChunks = useRef<Blob[]>([]);

  const audioStreamDestination = useRef<MediaStreamAudioDestinationNode>(audioContext.current.createMediaStreamDestination());
  const recordingMerger = useRef<ChannelMergerNode | null>(null);
  if (!recordingMerger.current) {
    audioStreamDestination.current.channelCount = 2;
    recordingMerger.current = audioContext.current.createChannelMerger(2);
    recordingMerger.current.connect(audioStreamDestination.current);
  }
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const audioRecorder = useRef<MediaRecorder>(new MediaRecorder(audioStreamDestination.current.stream, { mimeType: getMimeType("audio"), audioBitsPerSecond: 192000 }));
  const [videoURL, setVideoURL] = useState<string>("");
  const [audioURL, setAudioURL] = useState<string>("");
  const [isOver, setIsOver] = useState(false);
  const modelParams = useModelParams(params);
  const micDuration = useRef<number>(0);
  const micClock = useRef({ pageAt: 0, captureLatency: 0 });
  const actualAudioPlayed = useRef<number>(0);
  const textContainerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [userAnalyser, setUserAnalyser] = useState<AnalyserNode | null>(null);
  const [modelAnalyser, setModelAnalyser] = useState<AnalyserNode | null>(null);

  const WSURL = chatURL(modelParams.chatSessionId ?? "");

  const onDisconnect = useCallback(() => {
    setIsOver(true);
    stopRecording();
  }, [setIsOver]);

  const createSocket = useCallback(
    () => new RtcSocket(modelParams.chatSessionId ?? "", MIC_CONSTRAINTS, WSURL) as unknown as WebSocket,
    [modelParams.chatSessionId, WSURL],
  );
  const { isConnected, isWaitingForHandshake, sendMessage, socket, start, stop } = useSocket({
    uri: WSURL,
    onDisconnect,
    createSocket: useWebRtc ? createSocket : undefined,
  });
  useEffect(() => {
    audioRecorder.current.ondataavailable = (e) => {
      audioChunks.current.push(e.data);
    };
    audioRecorder.current.onstop = async () => {
      let blob: Blob;
      const mimeType = getMimeType("audio");
      if (mimeType.includes("webm")) {
        blob = await fixWebmDuration(new Blob(audioChunks.current, { type: mimeType }));
      } else {
        blob = new Blob(audioChunks.current, { type: mimeType });
      }
      setAudioURL(URL.createObjectURL(blob));
      audioChunks.current = [];
    };
  }, [mediaRecorder, audioRecorder, setVideoURL, setAudioURL, videoChunks, audioChunks]);


  useEffect(() => {
    start();
    return () => {
      stop();
    };
  }, [start]);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx) {
      // The video is laid out like this screen (white page): WaveLanes draws the name
      // and the lanes every frame, TextDisplay the transcript below them.
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = "white";
      ctx.fillRect(0, 0, VIDEO_SIZE, VIDEO_SIZE);
    }
  }, [canvasRef]);

  const startRecording = useCallback(() => {
    if (isRecording.current) {
      return;
    }
    if (canvasRef.current) {
      // Note: Attaching a track from this stream to the existing MediaRecorder
      // rather than creating a new MediaRecorder for the canvas stream
      // doesn't work on Safari as it just ends the recording immediately.
      // It works on Chrome though and is much cleaner.
      const captureStream = canvasRef.current.captureStream(30);
      captureStream.addTrack(audioStreamDestination.current.stream.getAudioTracks()[0]);
      mediaRecorder.current = new MediaRecorder(captureStream, { mimeType: getMimeType("video"), videoBitsPerSecond: 10000000, audioBitsPerSecond: 192000 });
      mediaRecorder.current.ondataavailable = (e) => {
        videoChunks.current.push(e.data);
      };
      mediaRecorder.current.onstop = async () => {
        let blob: Blob;
        const mimeType = getMimeType("video");
        if (mimeType.includes("webm")) {
          blob = await fixWebmDuration(new Blob(videoChunks.current, { type: mimeType }));
        } else {
          blob = new Blob(videoChunks.current, { type: mimeType });
        }
        setVideoURL(URL.createObjectURL(blob));
        videoChunks.current = [];
      };
    }
    worklet.current?.connect(recordingMerger.current!, 0, MODEL_CHANNEL);

    setVideoURL("");
    setAudioURL("");
    mediaRecorder.current?.start();
    audioRecorder.current.start();
    isRecording.current = true;
  }, [isRecording, setVideoURL, setVideoURL, worklet, audioStreamDestination, mediaRecorder, audioRecorder, canvasRef]);

  const stopRecording = useCallback(() => {
    if (!isRecording.current) {
      return;
    }
    worklet.current?.disconnect(recordingMerger.current!, 0, MODEL_CHANNEL);
    audioRecorder.current.stop();
    mediaRecorder.current?.stop();
    isRecording.current = false;
  }, [isRecording, worklet, audioStreamDestination, mediaRecorder, audioRecorder]);


  const restart = () => {
    localStorage.setItem("cfgScaleV4", modelParams.cfgScale.toString());
    document.location.reload();
  };
  // Which transport the conversation ended up on (WebRTC falls back to WebSocket).
  const transport = (socket as unknown as { transport?: string } | null)?.transport === "webrtc" ? "WebRTC" : "WebSocket";
  const statusText = isConnected ? `接続中（${transport}）` : isWaitingForHandshake ? "接続待ち…（ほかの会話が終わるのを待っています）" : isOver ? "会話は終了しました" : "切断";

  return (
    <SocketContext.Provider
      value={{
        isConnected,
        sendMessage,
        socket,
      }}
    >
      <MediaContext.Provider value={
        {
          startRecording,
          stopRecording,
          audioContext,
          worklet,
          audioStreamDestination,
          recordingMerger: recordingMerger as React.MutableRefObject<ChannelMergerNode>,
          micDuration,
          micClock,
          actualAudioPlayed,
        }
      }>
        <div className="flex h-screen flex-col gap-5 p-4 sm:p-6" style={{ height: "100dvh" }}>
          <div className="flex items-center gap-3">
            {isOver && (
              <Button accent onClick={restart}>設定に戻る</Button>
            )}
            {!isOver && !isWaitingForHandshake && (
              <Button
                onClick={() => {
                  audioContext.current.resume();
                  isConnected ? stop() : start();
                }}
              >
                {isConnected ? "会話を終える" : "接続する"}
              </Button>
            )}
            <span className={`h-2 w-2 shrink-0 rounded-full ${isConnected ? "bg-accent" : "bg-line"}`} />
            <span className="min-w-0 truncate text-[13px] text-muted">{statusText}</span>
            <div className="ml-auto shrink-0">
              <Masthead compact />
            </div>
          </div>

          {(audioURL || videoURL) && (
            <div className="flex flex-wrap items-center gap-2.5 rounded-[18px] bg-card p-[18px]">
              <span className="text-[13px] font-medium">この会話を保存</span>
              {audioURL && <a href={audioURL} download={`haru audio.${getExtension("audio")}`} className={capsuleClass({ accent: true })}>
                <AudioLines className="h-3.5 w-3.5" />音声
              </a>}
              {videoURL && <a href={videoURL} download={`haru video.${getExtension("video")}`} className={capsuleClass({ accent: true })}>
                <Film className="h-3.5 w-3.5" />動画
              </a>}
              {videoURL && getExtension("video") === "webm" &&
                <a href="https://restream.io/tools/webm-to-mp4-converter" target="_blank" rel="noreferrer"
                  className="text-[13px] text-accent hover:underline">mp4 に変換するには</a>}
            </div>
          )}

          <ServerAudioStats getAudioStats={getAudioStats} sessionId={modelParams.chatSessionId} />

          <ServerAudio
            onAnalyser={setModelAnalyser}
            setGetAudioStats={(callback: () => AudioStats) =>
              (getAudioStats.current = callback)
            }
          />
          <UserAudio onAnalyser={setUserAnalyser} />
          <WaveLanes
            userAnalyser={userAnalyser}
            modelAnalyser={modelAnalyser}
            userName={userName?.trim() || "あなた"}
            modelName={modelName?.trim() || "モデル"}
            copyCanvasRef={canvasRef}
          />

          <div className="min-h-[96px] flex-1 overflow-y-auto sm:pl-[126px]" ref={textContainerRef}>
            <TextDisplay containerRef={textContainerRef} copyCanvasRef={canvasRef} />
          </div>

          <canvas height={VIDEO_SIZE} width={VIDEO_SIZE} className="hidden" ref={canvasRef} />
        </div>
      </MediaContext.Provider>
    </SocketContext.Provider >
  );
};
