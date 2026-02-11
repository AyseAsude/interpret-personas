import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
const env = globalThis.process?.env;
const basePath = env?.VITE_BASE_PATH ?? "/";
const normalizedBasePath = basePath.endsWith("/") ? basePath : `${basePath}/`;
export default defineConfig({
    base: normalizedBasePath,
    plugins: [react()],
});
