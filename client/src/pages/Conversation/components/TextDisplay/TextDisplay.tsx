import { FC, useEffect, useRef, useState } from "react";
import { useServerText } from "../../hooks/useServerText";
import { FONT, Theme, VIDEO_LAYOUT, VIDEO_SCALE } from "../../../../haru/painters";

type TextDisplayProps = {
  containerRef: React.RefObject<HTMLDivElement>;
  copyCanvasRef?: React.RefObject<HTMLCanvasElement>;
};

// The last seconds of the model's words are dark, the rest grey.
const RECENT_MS = 2500;

// Transcript area of the hidden video canvas, in its 720-unit layout (below the lanes).
const VIDEO_TEXT = { x: 158, y: 350, width: 530, height: 300, fontSize: 26, lineHeight: 39 };
// Only the tail is laid out; it is far more than the area can show.
const VIDEO_TEXT_TAIL = 800;

type StyledChar = { c: string; recent: boolean };

// Intl.Segmenter is ES2022 (the project targets ES2020 libs).
type WordSegmenter = { segment: (s: string) => Iterable<{ segment: string }> };
const Segmenter = (Intl as unknown as {
  Segmenter?: new (locale: undefined, options: { granularity: "word" }) => WordSegmenter;
}).Segmenter;
const wordSegmenter = Segmenter ? new Segmenter(undefined, { granularity: "word" }) : null;

const segmentText = (s: string): string[] =>
  wordSegmenter ? Array.from(wordSegmenter.segment(s), seg => seg.segment) : Array.from(s);

// Draws the model's words like the on-page subtitles (tokens from `recentFrom` on dark,
// the rest grey), wrapped to the area and scrolled to the newest line.
export const drawVideoText = (
  ctx: CanvasRenderingContext2D,
  text: string[],
  recentFrom: number,
) => {
  const { x, y, width, height, fontSize, lineHeight } = VIDEO_TEXT;
  const chars: StyledChar[] = [];
  for (let i = text.length - 1; i >= 0 && chars.length < VIDEO_TEXT_TAIL; i--) {
    chars.unshift(...Array.from(text[i]).map(c => ({ c, recent: i >= recentFrom })));
  }
  ctx.save();
  ctx.setTransform(VIDEO_SCALE, 0, 0, VIDEO_SCALE, 0, 0);
  ctx.font = `500 ${fontSize}px ${FONT}`;
  const measure = (seg: StyledChar[]) => ctx.measureText(seg.map(ch => ch.c).join("")).width;

  const lines: StyledChar[][] = [[]];
  let lineWidth = 0;
  let pos = 0;
  for (const segment of segmentText(chars.map(ch => ch.c).join(""))) {
    const seg = chars.slice(pos, pos + Array.from(segment).length);
    pos += seg.length;
    if (segment.includes("\n")) {
      lines.push([]);
      lineWidth = 0;
      continue;
    }
    const isSpace = segment.trim() === "";
    let line = lines[lines.length - 1];
    if (isSpace && line.length === 0) {
      continue;
    }
    const segWidth = measure(seg);
    if (lineWidth + segWidth <= width) {
      line.push(...seg);
      lineWidth += segWidth;
      continue;
    }
    if (isSpace) {
      lines.push([]);
      lineWidth = 0;
      continue;
    }
    if (line.length > 0 && segWidth <= width) {
      lines.push([...seg]);
      lineWidth = segWidth;
      continue;
    }
    // A segment wider than the whole line: break it per character.
    for (const ch of seg) {
      const w = measure([ch]);
      if (lineWidth + w > width && line.length > 0) {
        line = [];
        lines.push(line);
        lineWidth = 0;
      }
      line.push(ch);
      lineWidth += w;
    }
  }

  ctx.fillStyle = Theme.paper;
  ctx.fillRect(0, y - 10, VIDEO_LAYOUT, VIDEO_LAYOUT - (y - 10));
  ctx.beginPath();
  ctx.rect(x, y, width, height);
  ctx.clip();
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  const visible = Math.floor(height / lineHeight);
  const first = Math.max(0, lines.length - visible);
  lines.slice(first).forEach((line, row) => {
    let cx = x;
    const baseline = y + row * lineHeight + fontSize;
    // Runs of one color are drawn together so that kerning stays as measured.
    let start = 0;
    while (start < line.length) {
      let end = start;
      while (end < line.length && line[end].recent === line[start].recent) end++;
      const run = line.slice(start, end).map(ch => ch.c).join("");
      ctx.fillStyle = line[start].recent ? Theme.ink : Theme.muted;
      ctx.fillText(run, cx, baseline);
      cx += ctx.measureText(run).width;
      start = end;
    }
  });
  ctx.restore();
};

// The model's words as they arrive: the last few seconds dark, the rest grey.
export const TextDisplay: FC<TextDisplayProps> = ({ containerRef, copyCanvasRef }) => {
  const { text } = useServerText();
  // Arrival time (performance.now()) of each token.
  const times = useRef<number[]>([]);
  const [, setTick] = useState(0);

  if (times.current.length > text.length) {
    times.current = [];
  }
  while (times.current.length < text.length) {
    times.current.push(performance.now());
  }
  const cutoff = performance.now() - RECENT_MS;
  let recentFrom = text.length;
  while (recentFrom > 0 && times.current[recentFrom - 1] > cutoff) {
    recentFrom--;
  }
  const hasRecent = recentFrom < text.length;

  // Re-render while some words are still "recent", so that they turn grey in time.
  useEffect(() => {
    if (!hasRecent) {
      return;
    }
    const id = window.setInterval(() => setTick(n => n + 1), 100);
    return () => window.clearInterval(id);
  }, [hasRecent]);

  useEffect(() => {
    containerRef.current?.scroll({ top: containerRef.current.scrollHeight, behavior: "smooth" });
  }, [text]);

  useEffect(() => {
    const ctx = copyCanvasRef?.current?.getContext("2d");
    if (ctx) {
      drawVideoText(ctx, text, recentFrom);
    }
  }, [text, recentFrom, copyCanvasRef]);

  return (
    <p className="whitespace-pre-wrap px-1 text-[22px] font-medium leading-[1.55]">
      {text.length === 0 ? <span className="text-muted">…</span> : <>
        <span className="text-muted">{text.slice(0, recentFrom).join("")}</span>
        <span className="text-ink">{text.slice(recentFrom).join("")}</span>
      </>}
    </p>
  );
};
