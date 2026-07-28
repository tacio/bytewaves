import { defineConfig } from "vite";
import { VitePWA } from "vite-plugin-pwa";

export default defineConfig({
  base: "./",
  plugins: [
    VitePWA({
      registerType: "autoUpdate",
      includeAssets: ["favicon.svg", "capture-worklet.js"],
      manifest: {
        name: "ByteWaves Acoustic Modem",
        short_name: "ByteWaves",
        description: "Send short text between devices using sound.",
        theme_color: "#07141a",
        background_color: "#07141a",
        display: "standalone",
        start_url: ".",
        icons: [{ src: "favicon.svg", sizes: "any", type: "image/svg+xml", purpose: "any maskable" }],
      },
      workbox: {
        globPatterns: ["**/*.{js,css,html,wasm,svg}"],
        navigateFallback: "index.html",
      },
    }),
  ],
});
