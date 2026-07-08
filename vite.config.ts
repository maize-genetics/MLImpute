import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const __dirname = dirname(fileURLToPath(import.meta.url));
const appVersion = JSON.parse(
  readFileSync(join(__dirname, "package.json"), "utf-8"),
).version as string;

// https://vitejs.dev/config/
export default defineConfig(({ mode, isPreview }) => {
  const isWeb = mode === "web";
  // Only proxy /docs during a real dev serve. `preview.proxy` defaults to
  // `server.proxy`, so leaving the proxy set would make `preview:web` forward
  // /docs to the MkDocs dev server instead of serving the built
  // dist-web/docs/ files (and the deployed static site has no proxy at all).
  const proxyDocsToMkDocs = isWeb && !isPreview;

  return {
    plugins: [react()],

    define: {
      __APP_VERSION__: JSON.stringify(appVersion),
    },

    // Prevent vite from obscuring rust errors
    clearScreen: false,

    server: {
      port: isWeb ? 3001 : 3000,
      strictPort: !isWeb,
      watch: {
        ignored: ["**/src-tauri/**"],
      },
      // In web-mode dev, proxy /docs/ to the local MkDocs server so the
      // Documentation card works without a full production build.
      // Start it with: pixi run -e docs docs-serve
      proxy: proxyDocsToMkDocs
        ? {
            "/docs": {
              target: "http://localhost:8000",
              changeOrigin: true,
              // MkDocs serves at its root, so strip the /docs prefix.
              rewrite: (path) => path.replace(/^\/docs/, ""),
            },
          }
        : undefined,
    },

    base: isWeb ? "./" : undefined,

    build: {
      ...(isWeb
        ? {
            outDir: "dist-web",
            rollupOptions: {
              external: [
                "@tauri-apps/api/core",
                "@tauri-apps/api/event",
                "@tauri-apps/plugin-dialog",
                "@tauri-apps/plugin-fs",
                "@tauri-apps/plugin-os",
                "@tauri-apps/plugin-shell",
              ],
            },
          }
        : {}),
    },
  };
});
