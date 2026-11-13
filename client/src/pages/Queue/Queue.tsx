import processorUrl from "../../audio-processor.ts?worker&url";
import { PlayoutProfile, setPlayoutProfile } from "../Conversation/playoutProfile";
import { FC, ReactNode, useEffect, useState, useCallback, useRef, MutableRefObject } from "react";
import { ChevronsUpDown, CircleCheck, LoaderCircle, TriangleAlert } from "lucide-react";
import { Conversation } from "../Conversation/Conversation";
import { loadTurnKey, saveTurnKey, TurnKey } from "../Conversation/turnKey";
import { Button, SwitchButton } from "../../components/Button/Button";
import { fieldClass } from "../../components/Input/Input";
import { Masthead } from "../../components/Haru/Masthead";
import { Stat, loadLastLatency } from "../Conversation/components/ServerAudio/ServerAudioStats";
import { useModelParams } from "../Conversation/hooks/useModelParams";
import { ReferenceAudio, ReferenceAudioValue } from "./ReferenceAudio";
import { CardTemplate, DialogueCard, parseCard, renderCard } from "./card";
import { errorMessage, isKey, setUiLanguage, storedLanguage, useT } from "../../i18n";

// One edited card per language, kept in this browser.
const cardKey = (lang: string) => lang === "ja" ? "dialogueCardV2" : `dialogueCardV2_${lang}`;
const REFERENCE_KEY = "referenceAudioV1";
type CardPreset = { id: string; card: DialogueCard };
type DefaultReference = Omit<ReferenceAudioValue, "name" | "previewUrl"> & { preview_url: string };

type ServerConfig = {
  languages: { code: string; label: string }[];
  default_language: string;
  card_templates: Record<string, CardTemplate>;
  card_presets: Record<string, CardPreset[]>;
  system_prompt_max_chars: number;
  reference_min_seconds: number;
  reference_max_seconds: number;
  time_limit_seconds: number;
  default_references: Record<string, DefaultReference>;
  webrtc?: boolean;
  playout?: Partial<PlayoutProfile>;
};

const defaultReference = (config: ServerConfig, lang: string): ReferenceAudioValue | undefined => {
  const entry = config.default_references[lang];
  if (!entry) {
    return undefined;
  }
  const { preview_url, ...rest } = entry;
  return { ...rest, name: "", previewUrl: preview_url };
};

// A bundled voice ("default" for the server's default language, else "default_<lang>"), as opposed to an upload or recording.
const isDefaultReferenceId = (id: string) => id === "default" || id.startsWith("default_");

function getFloatFromStorage(val: string | null) {
  return (val == null) ? undefined : parseFloat(val)
}

function loadStoredCard(lang: string): DialogueCard | null {
  try {
    const raw = localStorage.getItem(cardKey(lang));
    return raw ? parseCard(JSON.parse(raw)) : null;
  } catch {
    return null;
  }
}

function loadStoredReference(): ReferenceAudioValue | null {
  try {
    const raw = localStorage.getItem(REFERENCE_KEY);
    if (!raw) {
      return null;
    }
    const { id, name, seconds, trimmed } = JSON.parse(raw);
    return typeof id === "string" ? { id, name, seconds, trimmed } : null;
  } catch {
    return null;
  }
}

function storeReference(value: ReferenceAudioValue | null) {
  try {
    if (value) {
      const { id, name, seconds, trimmed } = value;
      localStorage.setItem(REFERENCE_KEY, JSON.stringify({ id, name, seconds, trimmed }));
    } else {
      localStorage.removeItem(REFERENCE_KEY);
    }
  } catch {
    // Storage may be unavailable (private mode); the page works without it.
  }
}

export const Queue: FC = () => {
  const t = useT();
  const [hasMicrophoneAccess, setHasMicrophoneAccess] = useState<boolean>(false);
  const [showMicrophoneAccessMessage, setShowMicrophoneAccessMessage] = useState<boolean>(false);
  const [, setShouldConnect] = useState<boolean>(false);
  const [backendReady, setBackendReady] = useState<boolean>(false);
  const [config, setConfig] = useState<ServerConfig | null>(null);
  // Language of the conversation: card template, presets and bundled voice (null until the config loads).
  const [language, setLanguageBase] = useState<string | null>(null);
  const [card, setCardBase] = useState<DialogueCard | null>(null);
  // Visitor's own Cloudflare TURN key for WebRTC (this browser only), for servers without TURN.
  const [turnKey, setTurnKeyBase] = useState<TurnKey>(() => loadTurnKey() ?? { keyId: "", token: "" });
  const setTurnKey = useCallback((value: TurnKey) => {
    setTurnKeyBase(value);
    saveTurnKey(value.keyId.trim() && value.token.trim() ? value : null);
  }, []);
  const hasTurnKey = !!(turnKey.keyId.trim() && turnKey.token.trim());
  const [reference, setReferenceBase] = useState<ReferenceAudioValue | null>(loadStoredReference);
  const [isStarting, setIsStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const modelParams = useModelParams({
    cfgScale: getFloatFromStorage(localStorage.getItem("cfgScaleV4")),
  });
  // Text of the CFG field, so a half-typed value ("1.") is not overwritten; follows the Settings slider.
  const [cfgInput, setCfgInput] = useState(modelParams.cfgScale.toString());
  useEffect(() => setCfgInput(modelParams.cfgScale.toString()), [modelParams.cfgScale]);

  // Latency of the conversation before this page load, if there was one.
  const [lastLatency] = useState(loadLastLatency);

  const audioContext = useRef<AudioContext | null>(null);
  const worklet = useRef<AudioWorkletNode | null>(null);
  useEffect(() => {
    let cancelled = false;
    const loadConfig = async () => {
      try {
        const response = await fetch("/api/config", { cache: "no-store" });
        const data = await response.json() as ServerConfig;
        if (!cancelled) {
          setConfig(data);
          setPlayoutProfile(data.playout);
          const stored = storedLanguage();
          const lang = stored && data.card_presets[stored] ? stored : data.default_language;
          setLanguageBase(lang);
          setUiLanguage(lang);
          setCardBase(loadStoredCard(lang) ?? parseCard(data.card_presets[lang][0].card));
          // Preselect the bundled voice of the language unless the viewer chose another one.
          setReferenceBase(current => (current == null || isDefaultReferenceId(current.id))
            ? defaultReference(data, lang) ?? null : current);
        }
      } catch (error) {
        console.error("Config fetch failed", error);
        if (!cancelled) {
          window.setTimeout(loadConfig, 2000);
        }
      }
    };
    loadConfig();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    const checkBackend = async () => {
      try {
        const response = await fetch("/api/ready", { cache: "no-store" });
        const status = await response.json() as { ready?: boolean };
        if (!cancelled) {
          setBackendReady(response.ok && status.ready === true);
        }
      } catch (error) {
        console.error("Backend readiness check failed", error);
        if (!cancelled) {
          setBackendReady(false);
        }
      }
    };

    checkBackend();
    const intervalId = window.setInterval(checkBackend, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, []);

  const setCardField = useCallback((key: keyof DialogueCard, value: string) => {
    if (!language) {
      return;
    }
    setCardBase(current => {
      const next = { ...(current ?? parseCard({})!), [key]: value } as DialogueCard;
      try {
        localStorage.setItem(cardKey(language), JSON.stringify(next));
      } catch {
        // ignore
      }
      return next;
    });
  }, [language]);

  // Replace every card field with a preset (kept like an edit, so a reload keeps it).
  const applyPreset = useCallback((preset: DialogueCard) => {
    const next = parseCard(preset);
    if (!next || !language) {
      return;
    }
    setCardBase(next);
    try {
      localStorage.setItem(cardKey(language), JSON.stringify(next));
    } catch {
      // ignore
    }
  }, [language]);
  const presets = config && language ? config.card_presets[language] ?? [] : [];
  const presetLabel = (id: string) => {
    const key = `preset.${id}`;
    return isKey(key) ? t(key) : id;
  };
  const example = (name?: string) => name ? t("field.example", { name }) : "";
  const sameCard = (a: DialogueCard | null, b: DialogueCard) => {
    const pb = parseCard(b);
    return !!a && !!pb && (Object.keys(pb) as (keyof DialogueCard)[]).every(k => (a[k] ?? "") === (pb[k] ?? ""));
  };

  const setReference = useCallback((value: ReferenceAudioValue | null) => {
    setReferenceBase(value);
    storeReference(value);
  }, []);

  // Switch the UI, the card (this language's own edits, else its first preset) and, unless
  // the viewer uploaded or recorded a voice, the bundled voice.
  const setLanguage = useCallback((lang: string) => {
    if (!config || !config.card_presets[lang]) {
      return;
    }
    setLanguageBase(lang);
    setUiLanguage(lang, true);
    setStartError(null);
    setCardBase(loadStoredCard(lang) ?? parseCard(config.card_presets[lang][0].card));
    if (reference == null || isDefaultReferenceId(reference.id)) {
      setReference(defaultReference(config, lang) ?? null);
    }
  }, [config, reference, setReference]);

  const getMicrophoneAccess = useCallback(async () => {
    try {
      await window.navigator.mediaDevices.getUserMedia({ audio: true });
      setHasMicrophoneAccess(true);
      return true;
    } catch (e) {
      console.error(e);
      setShowMicrophoneAccessMessage(true);
      setHasMicrophoneAccess(false);
    }
    return false;
  }, [setHasMicrophoneAccess, setShowMicrophoneAccessMessage]);

  const startProcessor = useCallback(async () => {
    if (!audioContext.current) {
      // The model's own rate, so the decoded Opus needs no resampling.
      audioContext.current = new AudioContext({ sampleRate: 24000 });
    }
    if (worklet.current) {
      return;
    }
    let ctx = audioContext.current;
    ctx.resume();
    try {
      worklet.current = new AudioWorkletNode(ctx, "haru-processor");
    } catch (err) {
      await ctx.audioWorklet.addModule(processorUrl);
      worklet.current = new AudioWorkletNode(ctx, "haru-processor");
    }
    worklet.current.connect(ctx.destination);
  }, [audioContext, worklet]);

  const template = config && language ? config.card_templates[language] : undefined;
  const promptText = card && template ? renderCard(card, template) : "";
  const missingPersona = !card || card.persona.trim() === "";
  const promptTooLong = !!config && promptText.length > config.system_prompt_max_chars;
  const canStart = backendReady && !!reference && !missingPersona && !promptTooLong && !isStarting;

  const onConnect = useCallback(async () => {
    if (!canStart || !reference) {
      return;
    }
    setIsStarting(true);
    setStartError(null);
    try {
      // Unlock audio inside the click handler (autoplay policies).
      await startProcessor();
      const response = await fetch("/api/session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          system_prompt: promptText,
          reference_audio_id: reference.id,
          guidance_scale: modelParams.cfgScale,
        }),
      });
      if (!response.ok) {
        if (response.status === 404) {
          setReference(null);
        }
        throw new Error(errorMessage(t, await response.text(), "error.start"));
      }
      const data = await response.json() as { id: string };
      modelParams.setChatSessionId(data.id);
      localStorage.setItem("cfgScaleV4", modelParams.cfgScale.toString());
      const hasAccess = await getMicrophoneAccess();
      if (hasAccess) {
        setShouldConnect(true);
      }
    } catch (e) {
      console.error(e);
      setStartError(e instanceof Error && e.message ? e.message : t("error.start"));
    } finally {
      setIsStarting(false);
    }
  }, [canStart, reference, promptText, modelParams, startProcessor, getMicrophoneAccess, setReference, t]);

  if (hasMicrophoneAccess && audioContext.current && worklet.current && modelParams.chatSessionId) {
    return (
      <Conversation
        audioContext={audioContext as MutableRefObject<AudioContext>}
        worklet={worklet as MutableRefObject<AudioWorkletNode>}
        useWebRtc={!!config?.webrtc || hasTurnKey}
        userName={card?.other_name}
        modelName={card?.name}
        chatSessionId={modelParams.chatSessionId}
        cfgScale={modelParams.cfgScale}
      />
    );
  }

  // Why the conversation cannot start yet, if anything.
  let blocker = "";
  if (!backendReady) {
    blocker = t("state.modelLoading");
  } else if (missingPersona) {
    blocker = t("state.personaMissing");
  } else if (config && promptTooLong) {
    blocker = t("error.prompt_too_long", { max_chars: config.system_prompt_max_chars });
  } else if (!reference) {
    blocker = t("state.voiceMissing");
  }
  const problem = startError ?? (showMicrophoneAccessMessage ? t("error.microphonePermission") : null);

  return (
    <div className="mx-auto flex w-full max-w-[720px] flex-col gap-5 px-4 py-7 sm:px-8">
      <div className="flex justify-center pb-2 pt-3">
        <Masthead />
      </div>

      <div className="flex flex-col items-end gap-2 pb-2">
        <div className="flex items-center justify-end gap-3.5">
          {isStarting || !backendReady ?
            <span className="flex items-center gap-1.5 text-xs text-muted">
              <LoaderCircle className="h-3.5 w-3.5 shrink-0 animate-spin" />{isStarting ? t("state.preparing") : blocker}
            </span>
            : blocker ?
              <span className="flex items-center gap-1.5 text-xs text-warning">
                <TriangleAlert className="h-3.5 w-3.5 shrink-0" />{blocker}
              </span>
              :
              <span className="flex items-center gap-1.5 text-xs text-accent">
                <CircleCheck className="h-3.5 w-3.5 shrink-0" />{t("state.ready")}
              </span>}
          <Button accent large disabled={!canStart} onClick={async () => await onConnect()} className="shrink-0">
            {isStarting ? t("state.connecting") : t("action.start")}
          </Button>
        </div>
        {problem && <p className="flex items-start gap-1.5 whitespace-pre-wrap text-xs text-warning">
          <TriangleAlert className="mt-px h-3.5 w-3.5 shrink-0" />{problem}
        </p>}
      </div>

      {lastLatency && <Card>
        <div className="flex items-end gap-7">
          <Stat big label={t("stats.lastLatency")} value={`${Math.round(lastLatency.p50 * 1000)} ms`} />
          <Stat label="p95" value={`${Math.round(lastLatency.p95 * 1000)} ms`} />
        </div>
      </Card>}

      <Card>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <SectionTitle title={t("character.title")} note={t("character.note")} />
          <div className="flex flex-wrap gap-1.5">
            {presets.map(p => (
              <Button key={p.id} selected={sameCard(card, p.card)} onClick={() => applyPreset(p.card)}>
                {presetLabel(p.id)}
              </Button>
            ))}
          </div>
        </div>
        {card && config && language ? <>
          <SelectField label={t("field.language")}
            value={language} disabled={isStarting} onChange={setLanguage}
            options={config.languages.map(l => [l.code, l.label] as [string, string])} />
          <div className="grid grid-cols-1 gap-x-3.5 gap-y-4 sm:grid-cols-2">
            <TextField label={t("field.modelName")} value={card.name} onChange={v => setCardField("name", v)} placeholder={example(presets[0]?.card.name)} />
            <TextField label={t("field.yourName")} value={card.other_name} onChange={v => setCardField("other_name", v)} placeholder={example(presets[0]?.card.other_name)} />
            <SelectField label={t("field.gender")} value={card.gender} onChange={v => setCardField("gender", v)}
              options={[["", t("option.unspecified")], ["female", t("option.female")], ["male", t("option.male")]]} />
            <SelectField label={t("field.backchannel")} value={card.backchannel} onChange={v => setCardField("backchannel", v)}
              options={[["", t("option.unspecified")], ["low", t("backchannel.low")], ["mid", t("backchannel.mid")], ["high", t("backchannel.high")]]} />
          </div>
          <AreaField label={t("field.persona")} rows={5} value={card.persona} onChange={v => setCardField("persona", v)} />
          <AreaField label={t("field.relationship")} rows={3} value={card.relationship} onChange={v => setCardField("relationship", v)} />
          <AreaField label={t("field.situation")} rows={3} value={card.situation} onChange={v => setCardField("situation", v)} />
          <p className="text-xs text-muted">{t("character.hint")}</p>
          <details className="text-[13px]">
            <summary className="cursor-pointer text-accent">{t("prompt.show")}</summary>
            <pre className="mt-2 whitespace-pre-wrap rounded-xl bg-paper p-3.5 font-[inherit] text-xs leading-relaxed">{promptText}</pre>
            <p className={`mt-1 text-right text-xs tabular-nums ${promptTooLong ? "text-warning" : "text-muted"}`}>
              {t("prompt.length", { count: promptText.length, max: config.system_prompt_max_chars })}
            </p>
          </details>
        </> : <p className="text-[13px] text-muted">{t("loading")}</p>}
      </Card>

      <Card>
        <SectionTitle title={t("voice.title")}
          note={t("voice.note", { min: config?.reference_min_seconds ?? 3, max: Math.round((config?.reference_max_seconds ?? 180) / 60) })} />
        <ReferenceAudio
          value={reference}
          onChange={setReference}
          defaultValue={config && language ? defaultReference(config, language) : undefined}
          defaultLabel={config?.languages.find(l => l.code === language)?.label}
          maxSeconds={config?.reference_max_seconds ?? 180}
          disabled={isStarting}
        />
      </Card>

      <Card>
        <SectionTitle title={t("settings.title")} note="" />
        {/* Each control sits between what its two ends do. */}
        <SettingRow label="CFG scale" help={t("settings.cfgHelp")}>
          <div className="flex min-w-0 max-w-full items-center gap-2.5">
            <Hint>{t("settings.cfgLow")}</Hint>
            <input type="range" className="w-[200px] min-w-0" step="0.05" min="1" max="2" disabled={isStarting}
              value={modelParams.cfgScale} onChange={e => modelParams.setCfgScale(parseFloat(e.target.value))} />
            <Hint>{t("settings.cfgHigh")}</Hint>
          </div>
          <input type="number" className={`${fieldClass} !w-20 !py-1 tabular-nums`} step="0.05" min="1" max="2"
            value={cfgInput} disabled={isStarting}
            onChange={e => {
              setCfgInput(e.target.value);
              const v = parseFloat(e.target.value);
              if (Number.isFinite(v) && v >= 1 && v <= 2) modelParams.setCfgScale(v);
            }}
            onBlur={() => setCfgInput(modelParams.cfgScale.toString())} />
        </SettingRow>
        <SwitchButton className="self-start" disabled={isStarting} onClick={modelParams.resetParams}>{t("settings.cfgReset")}</SwitchButton>

        <div className="flex flex-col gap-2.5 rounded-xl bg-paper p-3.5">
          <div className="flex items-center justify-between gap-3">
            <p className="text-[13px] font-semibold">{t("webrtc.title")}</p>
            {hasTurnKey && <Button onClick={() => setTurnKey({ keyId: "", token: "" })}>{t("webrtc.clear")}</Button>}
          </div>
          <p className="text-xs text-muted">
            {t("webrtc.help")}
          </p>
          <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
            <input type="text" className={fieldClass} value={turnKey.keyId} placeholder="TURN Key ID"
              autoComplete="off" spellCheck={false} disabled={isStarting}
              onChange={e => setTurnKey({ ...turnKey, keyId: e.target.value })} />
            <input type="password" className={fieldClass} value={turnKey.token} placeholder="TURN Key API Token"
              autoComplete="off" disabled={isStarting}
              onChange={e => setTurnKey({ ...turnKey, token: e.target.value })} />
          </div>
          <p className="text-xs text-muted">
            {t("webrtc.privacy")}
          </p>
        </div>
      </Card>

      <p className="pb-2 text-center text-xs text-muted">
        {t("footer.limit", { minutes: Math.round((config?.time_limit_seconds ?? 300) / 60) })} <a
          href="https://huggingface.co/higurashi-ai" className="text-accent hover:underline">@higurashi</a>
      </p>
    </div>
  )
};

// A grouped section: soft grey, rounded, like Settings.
const Card: FC<{ children: ReactNode }> = ({ children }) => (
  <section className="flex flex-col gap-4 rounded-[18px] bg-card p-[22px]">{children}</section>
);

const SectionTitle: FC<{ title: string, note: string }> = ({ title, note }) => (
  <div className="flex flex-col gap-[3px]">
    <h2 className="text-xl font-semibold">{title}</h2>
    {note && <p className="text-[13px] text-muted">{note}</p>}
  </div>
);

const Hint: FC<{ children: ReactNode }> = ({ children }) => <span className="text-xs text-muted">{children}</span>;

const SettingRow: FC<{ label: string, help?: string, children: ReactNode }> = ({ label, help, children }) => (
  <div className="flex flex-col gap-1">
    <div className="flex flex-wrap items-center gap-2.5">
      <span className="w-[84px] shrink-0 text-[13px] font-medium">{label}</span>
      {children}
    </div>
    {help && <p className="text-xs text-muted sm:pl-[94px]">{help}</p>}
  </div>
);

const FieldBox: FC<{ label: string, children: ReactNode }> = ({ label, children }) => (
  <label className="flex flex-col gap-1.5">
    <span className="text-xs font-medium text-muted">{label}</span>
    {children}
  </label>
);

const TextField: FC<{ label: string, value: string, onChange: (v: string) => void, placeholder?: string }> = ({ label, value, onChange, placeholder }) => (
  <FieldBox label={label}>
    <input type="text" className={fieldClass} value={value} placeholder={placeholder} onChange={e => onChange(e.target.value)} />
  </FieldBox>
);

const AreaField: FC<{ label: string, value: string, onChange: (v: string) => void, rows: number }> = ({ label, value, onChange, rows }) => (
  <FieldBox label={label}>
    <textarea className={`${fieldClass} leading-relaxed`} rows={rows} value={value} onChange={e => onChange(e.target.value)} />
  </FieldBox>
);

const SelectField: FC<{ label: string, value: string, onChange: (v: string) => void, options: [string, string][], disabled?: boolean }> = ({ label, value, onChange, options, disabled }) => (
  <FieldBox label={label}>
    <span className="relative block">
      <select className={`${fieldClass} appearance-none pr-9`} value={value} disabled={disabled} onChange={e => onChange(e.target.value)}>
        {options.map(([v, text]) => <option key={v} value={v}>{text}</option>)}
      </select>
      <ChevronsUpDown className="pointer-events-none absolute right-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted" />
    </span>
  </FieldBox>
);
