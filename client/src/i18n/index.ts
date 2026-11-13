// UI strings in the nine languages of the model. The UI language is the language
// chosen on the setup page; it is kept in this browser.
import { useSyncExternalStore } from "react";
import { en, Key, Strings } from "./en";
import { ja } from "./ja";
import { zh } from "./zh";
import { ko } from "./ko";
import { es } from "./es";
import { fr } from "./fr";
import { de } from "./de";
import { pt } from "./pt";
import { ru } from "./ru";

export type { Key };
type Params = Record<string, string | number>;
export type Translate = (key: Key, params?: Params) => string;

const tables: Record<string, Strings> = { en, ja, zh, ko, es, fr, de, pt, ru };
const LANGUAGE_KEY = "cardLanguageV1";

export const isKey = (key: string): key is Key => key in en;

// One translator per language, so `t` keeps its identity until the language changes.
const translators = new Map<string, Translate>();
const translator = (lang: string): Translate => {
  let t = translators.get(lang);
  if (!t) {
    const table = tables[lang] ?? en;
    t = (key, params) => {
      const text = table[key] ?? en[key];
      return params ? text.replace(/\{(\w+)\}/g, (match, name) => name in params ? String(params[name]) : match) : text;
    };
    translators.set(lang, t);
  }
  return t;
};

// The language chosen on an earlier visit, if any.
export const storedLanguage = (): string | null => {
  try {
    return localStorage.getItem(LANGUAGE_KEY);
  } catch {
    return null;
  }
};

let current = "en";
const listeners = new Set<() => void>();

// Switch every mounted component to `lang`; `remember` keeps it for the next visit.
export const setUiLanguage = (lang: string, remember = false) => {
  current = lang;
  document.documentElement.lang = lang;
  if (remember) {
    try {
      localStorage.setItem(LANGUAGE_KEY, lang);
    } catch {
      // Storage may be unavailable (private mode); the page works without it.
    }
  }
  listeners.forEach(listener => listener());
};

// Until the server's language list arrives, show the remembered language.
const remembered = storedLanguage();
if (remembered && tables[remembered]) {
  setUiLanguage(remembered);
}

const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
};

export const useT = (): Translate => translator(useSyncExternalStore(subscribe, () => current));

// Message of a failed API response: its `code` translated, else the server's own text.
export const errorMessage = (t: Translate, body: string, fallback: Key): string => {
  try {
    const { code, error, ...params } = JSON.parse(body);
    const key = `error.${code}`;
    return isKey(key) ? t(key, params) : error || t(fallback);
  } catch {
    return body || t(fallback);
  }
};
