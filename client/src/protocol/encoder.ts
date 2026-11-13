import { WSMessage } from "./types";

const KIND = { handshake: 0x00, audio: 0x01, text: 0x02, ping: 0x06 } as const;

export const encodeMessage = (message: WSMessage): Uint8Array => {
  switch (message.type) {
    case "handshake":
      return new Uint8Array([KIND.handshake]);
    case "audio":
      return new Uint8Array([KIND.audio, ...message.data]);
    case "text":
      return new Uint8Array([KIND.text, ...new TextEncoder().encode(message.data)]);
    case "ping":
      return new Uint8Array([KIND.ping]);
  }
};

// Unknown kinds decode as a ping (ignored by every listener).
export const decodeMessage = (data: Uint8Array): WSMessage => {
  const payload = data.slice(1);
  switch (data[0]) {
    case KIND.handshake:
      return { type: "handshake" };
    case KIND.audio:
      return { type: "audio", data: payload };
    case KIND.text:
      return { type: "text", data: new TextDecoder().decode(payload) };
    default:
      return { type: "ping" };
  }
};
