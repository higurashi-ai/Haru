// The visitor's own Cloudflare TURN key, entered in the UI and kept in this browser only.
// The browser exchanges it with Cloudflare for short-lived TURN credentials; the
// server only ever receives those credentials, never the key or its API token.

const TURN_KEY_STORAGE = "haruCloudflareTurnV1";
const CLOUDFLARE_TURN_URL = "https://rtc.live.cloudflare.com/v1/turn/keys/{id}/credentials/generate-ice-servers";
const TURN_TTL_SECONDS = 7200;  // longer than the longest conversation

export type TurnKey = { keyId: string; token: string };

export const loadTurnKey = (): TurnKey | null => {
  try {
    const stored = JSON.parse(localStorage.getItem(TURN_KEY_STORAGE) ?? "null");
    if (stored?.keyId && stored?.token) return { keyId: String(stored.keyId), token: String(stored.token) };
  } catch {
    // ignore
  }
  return null;
};

export const saveTurnKey = (key: TurnKey | null) => {
  try {
    if (key && key.keyId && key.token) localStorage.setItem(TURN_KEY_STORAGE, JSON.stringify(key));
    else localStorage.removeItem(TURN_KEY_STORAGE);
  } catch {
    // ignore
  }
};

// Short-lived ICE servers from Cloudflare; throws when the key is rejected or unreachable.
export const fetchTurnServers = async (key: TurnKey): Promise<RTCIceServer[]> => {
  const response = await fetch(CLOUDFLARE_TURN_URL.replace("{id}", encodeURIComponent(key.keyId.trim())), {
    method: "POST",
    headers: { Authorization: `Bearer ${key.token.trim()}`, "Content-Type": "application/json" },
    body: JSON.stringify({ ttl: TURN_TTL_SECONDS }),
  });
  if (!response.ok) throw new Error(`Cloudflare TURN: HTTP ${response.status}`);
  const payload = await response.json();
  const servers: any[] = Array.isArray(payload?.iceServers) ? payload.iceServers : [payload?.iceServers].filter(Boolean);
  const out: RTCIceServer[] = [];
  for (const server of servers) {
    const urls = (Array.isArray(server.urls) ? server.urls : [server.urls])
      .filter((u: unknown) => typeof u === "string" && !/:53(\?|$)/.test(u)); // browsers block port 53
    if (urls.length) out.push({ urls, username: server.username, credential: server.credential });
  }
  if (!out.some(s => s.username)) throw new Error("Cloudflare TURN: no relay credentials");
  return out;
};
