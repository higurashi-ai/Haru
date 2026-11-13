// Messages of the /api/chat WebSocket: one kind byte, then the payload.
export type WSMessage =
  | { type: "handshake" }
  | { type: "audio"; data: Uint8Array }  // Ogg/Opus pages, 24 kHz mono
  | { type: "text"; data: string }
  | { type: "ping" };
