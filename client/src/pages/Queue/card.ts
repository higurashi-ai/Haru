// Character card of the model speaker, rendered exactly as the server's
// ``render_card``: lead sentence and field headers in the card's language
// (templates from /api/config). The card is written from the model's point of view: "you" = the model.

export type Backchannel = "" | "low" | "mid" | "high";
export type Gender = "" | "female" | "male";

export type DialogueCard = {
  persona: string;
  name: string;
  gender: Gender;
  backchannel: Backchannel;
  other_name: string;
  relationship: string;
  situation: string;
};

export type CardTemplate = {
  lead: string;
  headers: Record<keyof DialogueCard, string>;
  gender: Record<string, string>;
  backchannel: Record<string, string>;
};

export const EMPTY_CARD: DialogueCard = {
  persona: "",
  name: "",
  gender: "",
  backchannel: "",
  other_name: "",
  relationship: "",
  situation: "",
};

// UI labels of the backchannel levels (the UI is Japanese whatever the card language).
export const BACKCHANNEL_TEXT: Record<string, string> = { low: "少なめ", mid: "普通", high: "多め" };

// Field order of the rendered card.
const CARD_FIELDS: (keyof DialogueCard)[] = ["persona", "name", "gender", "backchannel", "other_name", "relationship", "situation"];

export const renderCard = (card: DialogueCard, template: CardTemplate): string => {
  const lines = [template.lead];
  for (const key of CARD_FIELDS) {
    let text = (card[key] ?? "").trim();
    if (key === "gender" || key === "backchannel") {
      text = template[key][text] ?? "";
    }
    if (text) {
      lines.push(template.headers[key], text);
    }
  }
  return lines.join("\n");
};

export const parseCard = (value: unknown): DialogueCard | null => {
  if (!value || typeof value !== "object") {
    return null;
  }
  const source = value as Record<string, unknown>;
  const card = { ...EMPTY_CARD };
  for (const key of Object.keys(EMPTY_CARD) as (keyof DialogueCard)[]) {
    if (typeof source[key] === "string") {
      (card[key] as string) = source[key] as string;
    }
  }
  return card;
};
