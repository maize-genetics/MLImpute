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
export default defineConfig(({ mode }) => {
  const isWeb = mode === "web";

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
