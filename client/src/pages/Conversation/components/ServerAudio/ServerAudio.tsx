import { FC, useEffect } from "react";
import { AudioStats, useServerAudio } from "../../hooks/useServerAudio";
import { useT } from "../../../../i18n";

type ServerAudioProps = {
  setGetAudioStats: (getAudioStats: () => AudioStats) => void;
  // The analyser on the model's voice, for the waveform lane.
  onAnalyser: (analyser: AnalyserNode) => void;
};
export const ServerAudio: FC<ServerAudioProps> = ({ setGetAudioStats, onAnalyser }) => {
  const t = useT();
  const { analyser, hasCriticalDelay, setHasCriticalDelay } = useServerAudio({
    setGetAudioStats,
  });
  useEffect(() => {
    onAnalyser(analyser.current);
  }, [analyser, onAnalyser]);
  if (!hasCriticalDelay) {
    return null;
  }
  return (
    <div className="fixed left-0 top-0 z-50 flex w-screen items-center justify-between gap-3 bg-warning px-4 py-2 text-[13px] text-white">
      <p>{t("notice.reconnected")}</p>
      <button
        onClick={async () => {
          setHasCriticalDelay(false);
        }}
        className="rounded-full bg-white px-3 py-1 text-xs font-medium text-ink"
      >
        {t("notice.dismiss")}
      </button>
    </div>
  );
};
