import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const isWeb = mode === "web";

  return {
    plugins: [react()],

    // Prevent vite from obscuring rust errors
    clearScreen: false,

    server: {
      port: isWeb ? 3001 : 3000,
      strictPort: !isWeb,
      watch: {
        ignored: ["**/src-tauri/**"],
      },
    },

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
