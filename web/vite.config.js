import { defineConfig } from "vite";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  build: {
    outDir: resolve(here, "../server/static"),
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: resolve(here, "index.html"),
        extended: resolve(here, "extended.html"),
      },
    },
  },
});
