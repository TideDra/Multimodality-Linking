import { defineConfig } from "vite";
import path from "path";
import vue from "@vitejs/plugin-vue";
// https://github.com/vuetifyjs/vuetify-loader/tree/next/packages/vite-plugin
import vuetify from "vite-plugin-vuetify";
import AutoImport from "unplugin-auto-import/vite";

const pathSrc = path.resolve(__dirname, "src");

// https://vitejs.dev/config/
export default defineConfig({
  resolve: {
    alias: {
      "@": path.join(__dirname, "./src"),
      "~/": `${pathSrc}/`,
    },
  },

  plugins: [
    vue(),
    vuetify({ autoImport: true }),
    AutoImport({
      imports: ["vue"],
      dts: path.resolve(pathSrc, "auto-imports.d.ts"),
    }),
  ],
});
