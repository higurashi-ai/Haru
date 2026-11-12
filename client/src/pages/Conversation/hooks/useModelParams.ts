import { useCallback, useState } from "react";

// Guidance scale of the audio head (trained range 1..2).
export const DEFAULT_CFG_SCALE = 1.5;

export type ModelParamsValues = {
  cfgScale: number;
  // Conversation id registered with POST /api/session.
  chatSessionId: string | undefined;
};

export const useModelParams = (params?: Partial<ModelParamsValues>) => {
  const [cfgScale, setCfgScaleBase] = useState(params?.cfgScale || DEFAULT_CFG_SCALE);
  const [chatSessionId, setChatSessionId] = useState<string | undefined>(params?.chatSessionId);

  const resetParams = useCallback(() => setCfgScaleBase(DEFAULT_CFG_SCALE), []);
  const setCfgScale = useCallback((value: number) => {
    if (value <= 2.0 && value >= 1.0) {  // the server clamps the same
      setCfgScaleBase(value);
    }
  }, []);

  return { cfgScale, chatSessionId, setCfgScale, setChatSessionId, resetParams };
};
