import { defineConfig, loadEnv } from "vite";

// `npm run dev` serves the client with hot reload and forwards /api (HTTP and WebSocket)
// to a running `haru-server` (VITE_HARU_SERVER, default http://localhost:8998).
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd());
  return {
    server: {
      proxy: {
        "/api": { target: env.VITE_HARU_SERVER || "http://localhost:8998", changeOrigin: true, ws: true },
      },
    },
  };
});
