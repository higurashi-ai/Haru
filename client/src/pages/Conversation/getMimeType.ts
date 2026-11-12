// mp4 with H.264 video first (plays on iPhone, social media, slides) and Opus
// audio, the better codec at the same bitrate; AAC where the browser has no
// Opus-in-mp4 (Safari); webm (VP8/VP9 + Opus) where it cannot record mp4 at all.
const VIDEO_MIME_TYPES = [
  "video/mp4;codecs=avc1,opus",
  "video/mp4;codecs=avc1,mp4a.40.2",
  "video/mp4",
  "video/webm",
];
// Audio only: Opus in webm, else AAC in m4a.
const AUDIO_MIME_TYPES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/mp4;codecs=mp4a.40.2",
  "audio/mp4",
  "audio/mpeg",
];

const pickMimeType = (types: string[], fallback: string) => {
  if (!MediaRecorder.isTypeSupported) {
    return fallback;
  }
  return types.find(t => MediaRecorder.isTypeSupported(t)) ?? "";
};

const getVideoMimeType = () => pickMimeType(VIDEO_MIME_TYPES, "video/mp4");

const getAudioMimeType = () => pickMimeType(AUDIO_MIME_TYPES, "audio/mp4");

export const getMimeType = (type: "audio" | "video") => {
  if(type === "audio") {
    return getAudioMimeType();
  }
  return getVideoMimeType();
}

export const getExtension = (type: "audio" | "video") => {
  if(getMimeType(type).includes("mp4")) {
    return type === "audio" ? "m4a" : "mp4";
  }
  if(getMimeType(type).includes("mpeg")) {
    return "mp3";
  }
  return "webm";
}