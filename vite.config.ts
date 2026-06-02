/// <reference types="vitest/config" />
import { defineConfig } from 'vite';

// GitHub Pages serves this project site under /<repo-name>/, so the base path
// must match the repository name. Override with VITE_BASE for local/other hosts.
const base = process.env.VITE_BASE ?? '/neuroforge/';

export default defineConfig({
  base,
  build: {
    target: 'es2022',
    outDir: 'dist',
    sourcemap: false,
  },
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts', 'src/**/*.test.ts'],
    coverage: {
      provider: 'v8',
      include: ['src/**/*.ts'],
      exclude: ['src/main.ts', 'src/**/*.test.ts'],
    },
  },
});
