// Buffering of the browser side, from /api/config ("playout"; set by the server's
// --network-profile). Defaults = the "internet" profile.
export type PlayoutProfile = {
  // WebSocket playout: queued audio before playback starts, and the lowest queue the
  // catch-up shortens back to.
  playout_start_ms: number;
  catchup_target_ms: number;
  // 20 ms Opus frames per Ogg page sent from the mic.
  mic_frames_per_page: number;
};

let profile: PlayoutProfile = {
  playout_start_ms: 800 / 3,
  catchup_target_ms: 400 / 3 + 100,
  mic_frames_per_page: 2,
};

export const setPlayoutProfile = (value: Partial<PlayoutProfile> | undefined) => {
  if (value) {
    profile = { ...profile, ...value };
  }
};

export const getPlayoutProfile = () => profile;
