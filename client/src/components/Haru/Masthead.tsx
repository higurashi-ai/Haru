import { FC, useEffect, useRef } from "react";
import { BlobState, Palette, drawBlob, now } from "../../haru/painters";

// The model's orb at rest, as a logo.
export const IdleOrb: FC<{ size: number; className?: string }> = ({ size, className }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const state = new BlobState();
    let frame = 0;
    const draw = () => {
      frame = requestAnimationFrame(draw);
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");
      if (!canvas || !ctx) {
        return;
      }
      const dpr = window.devicePixelRatio || 1;
      if (canvas.width !== Math.round(size * dpr)) {
        canvas.width = canvas.height = Math.round(size * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, size, size);
      drawBlob(ctx, 0, 0, size, size, 0, now(), state, Palette.model);
    };
    draw();
    return () => cancelAnimationFrame(frame);
  }, [size]);
  return <canvas ref={canvasRef} className={className} style={{ width: size, height: size }} />;
};

export const Masthead: FC<{ compact?: boolean }> = ({ compact }) => (
  compact ?
    <div className="flex items-center gap-1.5">
      <IdleOrb size={30} />
      <span className="text-[22px] font-semibold tracking-[-0.3px]">Haru</span>
    </div>
    :
    <div className="flex flex-col items-center gap-1.5">
      <div className="flex items-center gap-2">
        {/* The orb fills about the middle half of its canvas: lay it out at that width. */}
        <IdleOrb size={84} className="-mx-5" />
        <h1 className="text-[52px] font-semibold leading-none tracking-[-1px]">Haru</h1>
      </div>
      <p className="text-xl font-medium text-muted">Full-Duplex Speech-to-Speech AI</p>
    </div>
);
