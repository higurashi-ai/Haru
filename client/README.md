# Haru web client

The browser UI of the Haru demo: character card, voice, settings, and the live
conversation view (waveforms, subtitles, latency, recording). React + Vite +
Tailwind. `haru-server` serves the built bundle from `haru/haru/static/`.

## Build

Node 20 or newer.

```bash
npm ci
npm run build                       # writes dist/
rm -rf ../haru/haru/static
cp -r dist ../haru/haru/static      # the bundle packaged with haru
```

To try a build without replacing the packaged one: `haru-server --static client/dist`.

## Develop

Start a server (`haru-server --hf-repo higurashi-ai/Haru-4B`), then

```bash
npm run dev
```

serves the client with hot reload and forwards `/api` (HTTP and WebSocket) to
`http://localhost:8998`. Set `VITE_HARU_SERVER` in `.env.local` for another
address. Microphone access needs a secure context: `localhost` or HTTPS.

## Layout

- `src/pages/Queue/` — setup page: language, character card, voice, guidance scale.
- `src/pages/Conversation/` — the conversation: transports (`hooks/useSocket.ts` for the
  WebSocket, `rtcSocket.ts` for WebRTC with WebSocket fallback), playback, statistics, recording.
- `src/audio-processor.ts` — the AudioWorklet that plays the model's audio (playout queue
  and catch-up after network stalls).
- `src/protocol/` — the binary messages of `/api/chat`.
- `public/assets/decoderWorker.min.*` — the Ogg/Opus decoder worker of
  [opus-recorder](https://github.com/chris-rudmin/opus-recorder).

## License

MIT, see [LICENSE](LICENSE). The client is derived from the web client of
[Moshi](https://github.com/kyutai-labs/moshi) by Kyutai.
