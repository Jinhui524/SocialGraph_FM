import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiTarget = (process.env.SOCIALGRAPH_E2E_API_ORIGIN
  ?? `http://127.0.0.1:${process.env.SOCIALGRAPH_CORE_API_PORT ?? "8000"}`).replace(/\/$/u, "");

const apiProxy = {
  "/api": {
    target: apiTarget,
    changeOrigin: true,
  },
};

export default defineConfig(() => ({
  define: { "import.meta.env.VITE_SOCIALGRAPH_API_BASE_URL": JSON.stringify("") },
  build: {
    outDir: "dist/client",
    chunkSizeWarningLimit: 1900,
  },
  optimizeDeps: {
    include: ["react", "react-dom/client"],
  },
  server: {
    host: "127.0.0.1",
    allowedHosts: ["terminal.local"],
    proxy: apiProxy,
    warmup: {
      clientFiles: ["./src/main.tsx"],
    },
  },
  preview: {
    host: "127.0.0.1",
    allowedHosts: ["terminal.local"],
    proxy: apiProxy,
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
  },
  plugins: [react()],
}));
