import { FC, RefObject, useEffect, useRef } from "react";
import {
  BlobState, FONT, MODEL, Palette, Theme, Track, USER, VIDEO_LAYOUT, VIDEO_SCALE, WaveHistory,
  drawBlob, drawLane, normalizedLevel, now, roundRect,
} from "../../../../haru/painters";

type WaveLanesProps = {
  userAnalyser: AnalyserNode | null;
  modelAnalyser: AnalyserNode | null;
  userName: string;
  modelName: string;
  // The hidden canvas recorded as the video.
  copyCanvasRef?: RefObject<HTMLCanvasElement>;
};

const ORB = 92, LANE_HEIGHT = 112;

// Layout of the video (720 units square): the name on
// top, your lane and the model's (the transcript below is drawn by TextDisplay).
const VIDEO = { padding: 32, logo: 34, laneX: 158, laneWidth: 530, userY: 88, modelY: 214, end: 340 };

// Peak of the newest `seconds` of what the analyser hears.
const readPeak = (analyser: AnalyserNode | null, buffer: { data: Float32Array<ArrayBuffer> | null }, seconds: number) => {
  if (!analyser) {
    return 0;
  }
  if (!buffer.data || buffer.data.length !== analyser.fftSize) {
    buffer.data = new Float32Array(analyser.fftSize);
  }
  const data = buffer.data;
  analyser.getFloatTimeDomainData(data);
  const count = Math.min(data.length, Math.max(128, Math.ceil(seconds * analyser.context.sampleRate)));
  let peak = 0;
  for (let i = data.length - count; i < data.length; i++) {
    const v = Math.abs(data[i]);
    if (v > peak) peak = v;
  }
  return peak;
};

// Sizes the canvas for the screen's pixel ratio and returns a context in CSS pixels.
const prepare = (canvas: HTMLCanvasElement | null) => {
  if (!canvas) {
    return null;
  }
  const width = canvas.clientWidth, height = canvas.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return null;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
};

// Both lanes, drawn from one clock reading per screen frame so they line up:
// your voice and the model's, newest at the right, each with a living orb on the left.
export const WaveLanes: FC<WaveLanesProps> = ({ userAnalyser, modelAnalyser, userName, modelName, copyCanvasRef }) => {
  const orbs = [useRef<HTMLCanvasElement>(null), useRef<HTMLCanvasElement>(null)];
  const lanes = [useRef<HTMLCanvasElement>(null), useRef<HTMLCanvasElement>(null)];
  const live = useRef({ userAnalyser, modelAnalyser, userName, modelName });
  live.current = { userAnalyser, modelAnalyser, userName, modelName };

  useEffect(() => {
    const history = new WaveHistory();
    const blobs = [new BlobState(), new BlobState()];
    const videoBlobs = [new BlobState(), new BlobState()];
    const logo = new BlobState();
    const buffers = [{ data: null as Float32Array<ArrayBuffer> | null }, { data: null as Float32Array<ArrayBuffer> | null }];
    let last = now();
    let frame = 0;

    const draw = () => {
      frame = requestAnimationFrame(draw);
      const t = now();
      const dt = Math.min(0.25, t - last);
      last = t;
      const { userAnalyser, modelAnalyser, userName, modelName } = live.current;
      const rows: { track: Track; analyser: AnalyserNode | null; colors: string[]; bars: string; name: string; videoY: number }[] = [
        { track: USER, analyser: userAnalyser, colors: Palette.user, bars: Palette.userBars, name: userName, videoY: VIDEO.userY },
        { track: MODEL, analyser: modelAnalyser, colors: Palette.model, bars: Palette.modelBars, name: modelName, videoY: VIDEO.modelY },
      ];
      const video = copyCanvasRef?.current?.getContext("2d") ?? null;
      if (video) {
        video.setTransform(VIDEO_SCALE, 0, 0, VIDEO_SCALE, 0, 0);
        video.fillStyle = Theme.paper;
        video.fillRect(0, 0, VIDEO_LAYOUT, VIDEO.end);
        drawBlob(video, VIDEO.padding, VIDEO.padding, VIDEO.logo, VIDEO.logo, 0, t, logo, Palette.model);
        video.fillStyle = Theme.ink;
        video.font = `600 26px ${FONT}`;
        video.textAlign = "left";
        video.textBaseline = "middle";
        video.fillText("Haru", VIDEO.padding + VIDEO.logo + 6, VIDEO.padding + VIDEO.logo / 2 + 1);
      }
      rows.forEach((row, i) => {
        const peak = readPeak(row.analyser, buffers[i], dt);
        history.add(row.track, t, peak);
        const level = normalizedLevel(peak);
        const orb = prepare(orbs[i].current);
        if (orb) {
          drawBlob(orb.ctx, 0, 0, orb.width, orb.height, level, t, blobs[i], row.colors);
        }
        const lane = prepare(lanes[i].current);
        if (lane) {
          drawLane(lane.ctx, 0, 0, lane.width, lane.height, history, row.track, t, row.bars);
        }
        if (video) {
          drawBlob(video, VIDEO.padding + 9, row.videoY, ORB, ORB, level, t, videoBlobs[i], row.colors);
          video.fillStyle = Theme.muted;
          video.font = `500 12px ${FONT}`;
          video.textAlign = "center";
          video.textBaseline = "alphabetic";
          video.fillText(row.name, VIDEO.padding + 55, row.videoY + ORB + 16, 106);
          roundRect(video, VIDEO.laneX, row.videoY, VIDEO.laneWidth, LANE_HEIGHT, 22);
          video.fillStyle = Palette.lane;
          video.fill();
          video.save();
          video.translate(VIDEO.laneX, 0);
          drawLane(video, 0, row.videoY, VIDEO.laneWidth, LANE_HEIGHT, history, row.track, t, row.bars);
          video.restore();
        }
      });
    };
    draw();
    return () => cancelAnimationFrame(frame);
  }, [copyCanvasRef]);

  return (
    <div className="flex flex-col gap-3.5">
      {[userName, modelName].map((name, i) => (
        <div key={i} className="flex items-center gap-2 sm:gap-4">
          <div className="flex w-[84px] shrink-0 flex-col items-center gap-1 sm:w-[110px]">
            <canvas ref={orbs[i]} style={{ width: ORB, height: ORB }} />
            <span className="max-w-full truncate text-xs font-medium text-muted">{name}</span>
          </div>
          <canvas ref={lanes[i]} className="min-w-0 flex-1 rounded-[22px] bg-lane" style={{ height: LANE_HEIGHT }} />
        </div>
      ))}
    </div>
  );
};
